# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Main dependencies
import numpy
import scipy.fftpack


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
                
        Example:
        
            print(fftind(5))
            
            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]
            
        """
    k_ind = numpy.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )



def gaussian_random_field(alpha = 3.0,
                          size = 128, 
                          flag_normalize = True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            alpha (double, default = 3.0): 
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0

        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field
                
        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = numpy.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = numpy.random.normal(size = (size, size)) \
        + 1j * numpy.random.normal(size = (size, size))
    
        # To real space
    gfield = numpy.fft.ifft2(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - numpy.mean(gfield)
        gfield = gfield/numpy.std(gfield)
        
    return gfield

import torch
def get_guassian_2d_rand_mask(grid_size, noise_patch_size, thresh = None):
    guassian_map = gaussian_random_field(size=grid_size, alpha = 4)
    guassian_map = torch.tensor(guassian_map)
    # cur_thresh = thresh_range[global_step]
    if thresh is None:
        cur_thresh = torch.randn(1).item()
    else:
        cur_thresh = thresh
    rand_mask = guassian_map > cur_thresh
    rand_mask = rand_mask.repeat_interleave(noise_patch_size, dim = -1).repeat_interleave(noise_patch_size, dim = -2)
    return rand_mask

dilate_kernel_size = 5
kernel = numpy.ones([dilate_kernel_size, dilate_kernel_size])
dilate_kernel_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(kernel, 0), 0))
blur_kernel_size = 5
kernel = numpy.ones([blur_kernel_size, blur_kernel_size]) / (blur_kernel_size ** 2)
blur_kernel_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(kernel, 0), 0))

def set_smooth_kernel(blur_ks = 5, dilate_ks = 5):
    global dilate_kernel_tensor, blur_kernel_tensor, blur_kernel_size, dilate_kernel_size
    blur_kernel_size = blur_ks
    dilate_kernel_size = dilate_ks
    kernel = numpy.ones([dilate_kernel_size, dilate_kernel_size])
    dilate_kernel_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(kernel, 0), 0))
    kernel = numpy.ones([blur_kernel_size, blur_kernel_size]) / (blur_kernel_size ** 2)
    blur_kernel_tensor = torch.Tensor(numpy.expand_dims(numpy.expand_dims(kernel, 0), 0))


def dilate(im_tensor):
    dtype = im_tensor.dtype
    result = torch.clamp(torch.nn.functional.conv2d(im_tensor, dilate_kernel_tensor, padding=(dilate_kernel_size // 2, dilate_kernel_size // 2)), 0, 1).int()
    return result.to(dtype)

def blur(im_tensor):
    result = torch.clamp(torch.nn.functional.conv2d(im_tensor, blur_kernel_tensor, padding=(blur_kernel_size // 2, blur_kernel_size // 2)), 0, 1)
    return result

def get_rand_masks(batch_size, grid_size, thresh = None, noise_patch_size = 1, smooth = False, dtype = torch.float32):
    # rand_grid_size = torch.randint(grid_size_range[0], grid_size_range[1], (bsz,))
    grid_sizes = [grid_size] * batch_size
    rand_masks = [get_guassian_2d_rand_mask(gs, noise_patch_size, thresh = thresh).view(1,1,grid_size, grid_size).to(dtype) for gs in grid_sizes]

    # rand_masks = [F.interpolate(rand_mask, size = mask_size, mode="bilinear") for rand_mask in rand_masks]
    # kernel_sizes = [1,3,5,7,9,11,13,15,17]
    # rand_masks[0] = blur(rand_masks[0].view(1,1,*rand_masks[0].shape), 5)
    # rand_masks[0] = rand_masks[0].view(1,1,*rand_masks[0].shape)
    # rand_masks = [dilate(rand_masks[0], kernel_size) for kernel_size in kernel_sizes]
    rand_masks = torch.cat(rand_masks, dim = 0)
    if smooth:
        rand_masks = dilate(blur(rand_masks))
        one_minus_rand_masks = 1 - rand_masks
        rand_tensor = (torch.rand(batch_size) < 0.5).view(-1, 1, 1, 1)
        rand_masks = torch.where(rand_tensor, rand_masks, one_minus_rand_masks) 
        # rand_masks = blur(dilate(rand_masks, dilate_kernel), blur_kernel).int().to(torch.float32)

    return rand_masks

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field(size=16, alpha = 4)
    example = example > -2.2799279927992
    plt.imshow(example, cmap='gray')
    plt.savefig("tmp.png")
    
if __name__ == '__main__':
    main()

