import math
import time
from typing import Type, Dict, Any, Tuple, Callable
import copy
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F

from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper

from operator import attrgetter
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import BaseOutput
from diffusers.utils import logging
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from peft.tuners.lora.layer import Linear, BaseTunerLayer

logger = logging.get_logger(__name__) 

class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def lora_forward_hack(self):
    def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                if "use_lora" in kwargs and active_adapter not in kwargs["use_lora"]:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result_lora = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result_lora = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

                lora_mask = self.lora_mask[active_adapter]
                lora_mask = lora_mask.repeat_interleave(x.shape[0] // len(lora_mask), dim=0)
                result[lora_mask] = result_lora[lora_mask]

            result = result.to(torch_result_dtype)
        return result
    return forward

def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """


    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        enable_joint_attention = True

        # def __init__(self, *args, **kwargs) -> None:
        #     super().__init__(*args, **kwargs)
        #     self.attn1n = copy.deepcopy(self.attn1)
        #     self.norm1n = copy.deepcopy(self.norm1)
        def set_joint_layer_requires_grad(self, adapter_names, requires_grad):
            
            for module in self.attn1n.modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                if isinstance(adapter_names, str):
                    adapter_names = [adapter_names]

                # Deactivate grads on the inactive adapter and activate grads on the active adapter
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in adapter_names:
                            # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                            # happen if a completely different adapter layer is being activated.
                            layer.requires_grad_(requires_grad)

            # self.norm1n.requires_grad_(requires_grad)
            self.conv1n.requires_grad_(requires_grad)

        def initialize_joint_layers(self):
            self.attn1n = copy.deepcopy(self.attn1)
            # self.norm1n = copy.deepcopy(self.norm1)
            conv1n = nn.Linear(self.attn1n.out_dim, self.attn1n.out_dim, bias=False)
            self.conv1n = zero_module(conv1n)

        def set_joint_attention(self, enable = True):
            self.enable_joint_attention = enable

        # def hack_lora_forward(self):
        #     for name, module in self.named_modules():
        #         if isinstance(module, Linear):
        #             module.forward = lora_forward_hack(module)

        def initialize_joint_lora(self, adapter_name, joint_adapter_name):
            for name, module in self.attn1n.named_modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                attr = attrgetter(name)
                attn1_module = attr(self.attn1)
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    attn1_module_dict = getattr(attn1_module, layer_name)
                    for key, layer in module_dict.items():
                        if key == joint_adapter_name:
                            attn1_state_dict = attn1_module_dict[adapter_name].state_dict()
                            layer.load_state_dict(attn1_state_dict)


        # def set_lora_mask(self, lora_name, lora_mask):
        #     if not isinstance(lora_mask, torch.Tensor):
        #         lora_mask = torch.tensor(lora_mask, dtype=torch.bool)
        #     for name, module in self.named_modules():
        #         if isinstance(module, Linear):
        #             if not hasattr(module, "lora_mask"):
        #                 module.lora_mask = dict()
        #             if "attn1n.to_k" in name or "attn1n.to_v" in name:
        #                 module.lora_mask[lora_name] = ~lora_mask
        #             else:
        #                 module.lora_mask[lora_name] = lora_mask


        # def forward(
        #     self,
        #     hidden_states: torch.FloatTensor,
        #     attention_mask: Optional[torch.FloatTensor] = None,
        #     encoder_hidden_states: Optional[torch.FloatTensor] = None,
        #     encoder_attention_mask: Optional[torch.FloatTensor] = None,
        #     timestep: Optional[torch.LongTensor] = None,
        #     cross_attention_kwargs: Dict[str, Any] = None,
        #     class_labels: Optional[torch.LongTensor] = None,
        #     added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        # ) -> torch.FloatTensor:
        #     if cross_attention_kwargs is not None:
        #         if cross_attention_kwargs.get("scale", None) is not None:
        #             logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

        #     # Notice that normalization is always applied before the real computation in the following blocks.
        #     # 0. Self-Attention
        #     batch_size = hidden_states.shape[0]

        #     if self.norm_type == "ada_norm":
        #         norm_hidden_states = self.norm1(hidden_states, timestep)
        #     elif self.norm_type == "ada_norm_zero":
        #         norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        #             hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        #         )
        #     elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        #         norm_hidden_states = self.norm1(hidden_states)
        #     elif self.norm_type == "ada_norm_continuous":
        #         norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        #     elif self.norm_type == "ada_norm_single":
        #         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #             self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        #         ).chunk(6, dim=1)
        #         norm_hidden_states = self.norm1(hidden_states)
        #         norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        #         norm_hidden_states = norm_hidden_states.squeeze(1)
        #     else:
        #         raise ValueError("Incorrect norm used")

        #     if self.pos_embed is not None:
        #         norm_hidden_states = self.pos_embed(norm_hidden_states)

        #     # 1. Prepare GLIGEN inputs
        #     cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        #     gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        #     attn_output = self.attn1(
        #         norm_hidden_states,
        #         encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        #         attention_mask=attention_mask,
        #         **cross_attention_kwargs,
        #     )
        #     if self.norm_type == "ada_norm_zero":
        #         attn_output = gate_msa.unsqueeze(1) * attn_output
        #     elif self.norm_type == "ada_norm_single":
        #         attn_output = gate_msa * attn_output

        #     hidden_states = attn_output + hidden_states
        #     if hidden_states.ndim == 4:
        #         hidden_states = hidden_states.squeeze(1)

        #     # 1.2 GLIGEN Control
        #     if gligen_kwargs is not None:
        #         hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        #     # Joint Dual Cross-Attention
        #     if self.norm_type == "ada_norm":
        #         norm_hidden_states = self.norm1n(hidden_states, timestep)
        #     elif self.norm_type == "ada_norm_zero":
        #         norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1n(
        #             hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        #         )
        #     elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        #         norm_hidden_states = self.norm1n(hidden_states)
        #     elif self.norm_type == "ada_norm_continuous":
        #         norm_hidden_states = self.norm1n(hidden_states, added_cond_kwargs["pooled_text_emb"])
        #     elif self.norm_type == "ada_norm_single":
        #         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #             self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        #         ).chunk(6, dim=1)
        #         norm_hidden_states = self.norm1n(hidden_states)
        #         norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        #         norm_hidden_states = norm_hidden_states.squeeze(1)
        #     else:
        #         raise ValueError("Incorrect norm used")

        #     if self.pos_embed is not None:
        #         norm_hidden_states = self.pos_embed(norm_hidden_states)
            
        #     batch_size = norm_hidden_states.shape[0]

        #     lora_mask = next(iter(self.attn1n.to_q.lora_mask.values()))
        #     lora_mask = lora_mask.repeat_interleave(batch_size // len(lora_mask), dim=0)
        #     joint_encoder_hidden_states = torch.empty_like(norm_hidden_states)
        #     joint_encoder_hidden_states[~lora_mask] = norm_hidden_states[lora_mask]
        #     joint_encoder_hidden_states[lora_mask] = norm_hidden_states[~lora_mask]


        #     attn_output = self.attn1n(
        #         norm_hidden_states,
        #         encoder_hidden_states=joint_encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         **cross_attention_kwargs,
        #     )
        #     if self.norm_type == "ada_norm_zero":
        #         attn_output = gate_msa.unsqueeze(1) * attn_output
        #     elif self.norm_type == "ada_norm_single":
        #         attn_output = gate_msa * attn_output

        #     attn_output = self.conv1n(attn_output)
        #     hidden_states = attn_output + hidden_states
        #     if hidden_states.ndim == 4:
        #         hidden_states = hidden_states.squeeze(1)








        #     # 3. Cross-Attention
        #     if self.attn2 is not None:
        #         if self.norm_type == "ada_norm":
        #             norm_hidden_states = self.norm2(hidden_states, timestep)
        #         elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
        #             norm_hidden_states = self.norm2(hidden_states)
        #         elif self.norm_type == "ada_norm_single":
        #             # For PixArt norm2 isn't applied here:
        #             # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
        #             norm_hidden_states = hidden_states
        #         elif self.norm_type == "ada_norm_continuous":
        #             norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        #         else:
        #             raise ValueError("Incorrect norm")

        #         if self.pos_embed is not None and self.norm_type != "ada_norm_single":
        #             norm_hidden_states = self.pos_embed(norm_hidden_states)

        #         attn_output = self.attn2(
        #             norm_hidden_states,
        #             encoder_hidden_states=encoder_hidden_states,
        #             attention_mask=encoder_attention_mask,
        #             **cross_attention_kwargs,
        #         )
        #         hidden_states = attn_output + hidden_states

        #     # 4. Feed-forward
        #     # i2vgen doesn't have this norm 🤷‍♂️
        #     if self.norm_type == "ada_norm_continuous":
        #         norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        #     elif not self.norm_type == "ada_norm_single":
        #         norm_hidden_states = self.norm3(hidden_states)

        #     if self.norm_type == "ada_norm_zero":
        #         norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        #     if self.norm_type == "ada_norm_single":
        #         norm_hidden_states = self.norm2(hidden_states)
        #         norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        #     if self._chunk_size is not None:
        #         # "feed_forward_chunk_size" can be used to save memory
        #         ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        #     else:
        #         ff_output = self.ff(norm_hidden_states)

        #     if self.norm_type == "ada_norm_zero":
        #         ff_output = gate_mlp.unsqueeze(1) * ff_output
        #     elif self.norm_type == "ada_norm_single":
        #         ff_output = gate_mlp * ff_output

        #     hidden_states = ff_output + hidden_states
        #     if hidden_states.ndim == 4:
        #         hidden_states = hidden_states.squeeze(1)

        #     return hidden_states
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.FloatTensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)



            if self.enable_joint_attention:
                batch_size = norm_hidden_states.shape[0]

                lora_mask = self.attn1n.to_q.lora_mask["xy_lora"]
                height, width = lora_mask.shape[-2:]
                downsample_rate = int(np.sqrt((height * width) // norm_hidden_states.shape[1]))
                lora_mask = lora_mask[::downsample_rate, ::downsample_rate]
                lora_mask = lora_mask.view(-1)


                attn_output = torch.empty_like(norm_hidden_states)
                cross_attention_kwargs["use_lora"] = ["y_lora"]

                attn_output[:, lora_mask, :] = self.attn1(
                    norm_hidden_states[:, lora_mask, :],
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                cross_attention_kwargs["use_lora"] = ["x_lora"]
                attn_output[~lora_mask] = self.attn1(
                    norm_hidden_states[~lora_mask],
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                joint_encoder_hidden_states = torch.empty_like(norm_hidden_states)
                joint_encoder_hidden_states[~lora_mask] = norm_hidden_states[lora_mask]
                joint_encoder_hidden_states[lora_mask] = norm_hidden_states[~lora_mask]

                attn_output1n = self.attn1n(
                    norm_hidden_states,
                    encoder_hidden_states=joint_encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                attn_output1n = self.conv1n(attn_output1n)

                # attn_output[~lora_mask] = attn_output1n[~lora_mask]
                attn_output = attn_output + attn_output1n
                # attn_output = attn_output1n
            else:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )


            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])


            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            # i2vgen doesn't have this norm 🤷‍♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states            

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):
    """ Adds a forward pre hook to initialize random number generator.
        All modules share the same generator state to keep their randomness in VidToMe consistent in one pass.
        This hook can be removed with remove_patch. """
    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(
                args[0].device, fallback=module.generator)
        else:
            return None

        # module.generator = module.generator.manual_seed(module._tome_info["args"]["seed"])
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        local_merge_ratio: float = 0.9,
        merge_global: bool = False,
        global_merge_ratio=0.8,
        max_downsample: int = 2,
        seed: int = 123,
        batch_size: int = 2,
        include_control: bool = False,
        align_batch: bool = False,
        target_stride: int = 4,
        global_rand=0.5):
    """
    Patches a stable diffusion model with VidToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - local_merge_ratio: The ratio of tokens to merge locally. I.e., 0.9 would merge 90% src tokens.
              If there are 4 frames in a chunk (3 src, 1 dst), the compression ratio will be 1.3 / 4.0.
              And the largest compression ratio is 0.25 (when local_merge_ratio = 1.0).
              Higher values result in more consistency, but with more visual quality loss.
     - merge_global: Whether or not to include global token merging.
     - global_merge_ratio: The ratio of tokens to merge locally. I.e., 0.8 would merge 80% src tokens.
                           When find significant degradation in video quality. Try to lower the value.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply VidToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - seed: Manual random seed. 
     - batch_size: Video batch size. Number of video chunks in one pass. When processing one video, it 
                   should be 2 (cond + uncond) or 3 (when using PnP, source + cond + uncond).
     - include_control: Whether or not to patch ControlNet model.
     - align_batch: Whether or not to align similarity matching maps of samples in the batch. It should
                    be True when using PnP as control.
     - target_stride: Stride between target frames. I.e., when target_stride = 4, there is 1 target frame
                      in any 4 consecutive frames. 
     - global_rand: Probability in global token merging src/dst split. Global tokens are always src when
                    global_rand = 1.0 and always dst when global_rand = 0.0 .
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(
        model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError(
                "Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
        diffusion_models = [diffusion_model, model.controlnet]
    else:
        diffusion_models = [diffusion_model]

    for diffusion_model in diffusion_models:
        diffusion_model._tome_info = {
            "size": None,
            "hooks": [],
            "args": {
                "max_downsample": max_downsample,
                "generator": None,
                "seed": seed,
                "batch_size": batch_size,
                "align_batch": align_batch,
                "merge_global": merge_global,
                "global_merge_ratio": global_merge_ratio,
                "local_merge_ratio": local_merge_ratio,
                "global_rand": global_rand,
                "target_stride": target_stride
            }
        }
        hook_tome_model(diffusion_model)

        for name, module in diffusion_model.named_modules():
            # If for some reason this has a different name, create an issue and I'll fix it
            # if isinstance_str(module, "BasicTransformerBlock") and "down_blocks" not in name:
            if isinstance_str(module, "BasicTransformerBlock"):
                make_tome_block_fn = make_diffusers_tome_block
                module.__class__ = make_tome_block_fn(module.__class__)
                module._tome_info = diffusion_model._tome_info
                hook_tome_module(module)

                # Something introduced in SD 2.0 (LDM only)
                if not hasattr(module, "disable_self_attn") and not is_diffusers:
                    module.disable_self_attn = False

                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers

    model = model.unet if hasattr(model, "unet") else model
    model_ls = [model]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._tome_info["hooks"].clear()

            if module.__class__.__name__ == "ToMeBlock":
                module.__class__ = module._parent

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for k, v in kwargs.items():
                    setattr(module, k, v)
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """ Collect attributes in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res

    return ret_dict

def set_patch_lora_mask(model: torch.nn.Module, lora_name, lora_mask):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)

    lora_mask = torch.tensor(lora_mask, dtype=torch.bool)
    for model in model_ls:
        if not hasattr(model, "lora_mask"):
            model.lora_mask = dict()
        model.lora_mask[lora_name] = lora_mask
        
        for name, module in model.named_modules():
            # if module.__class__.__name__ == "ToMeBlock":
            #     module.set_lora_mask(lora_name, lora_mask_map)
            if isinstance(module, Linear):
                if not hasattr(module, "lora_mask"):
                    module.lora_mask = dict()
                if "attn1n.to_k" in name or "attn1n.to_v" in name:
                    module.lora_mask[lora_name] = ~lora_mask
                else:
                    module.lora_mask[lora_name] = lora_mask
    return model

def set_joint_layer_requires_grad(model: torch.nn.Module, adapter_names, requires_grad):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if module.__class__.__name__ == "ToMeBlock":
                module.set_joint_layer_requires_grad(adapter_names, requires_grad)
    return model

def hack_lora_forward(model: torch.nn.Module):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                module.forward = lora_forward_hack(module)
    return model


def initialize_joint_lora(model: torch.nn.Module, adapter_name, joint_adapter_name):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if module.__class__.__name__ == "ToMeBlock":
                module.initialize_joint_lora(adapter_name, joint_adapter_name)
    return model

def set_joint_attention(model: torch.nn.Module, enable = True):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if module.__class__.__name__ == "ToMeBlock":
                module.set_joint_attention(enable = enable)
    return model

def initialize_joint_layers(model: torch.nn.Module):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if module.__class__.__name__ == "ToMeBlock":
                module.initialize_joint_layers()
    return model

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
