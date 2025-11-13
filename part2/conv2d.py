import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height*out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax   = nl.tile_size.pmax
    c_out_pmax  = nl.tile_size.pmax # FIXME check whether this is true
    h_tile_size = 2 # FIXME need to increase this and account for boundary conditions

    n_tiles_c_in    = in_channels // c_in_pmax
    n_tiles_c_out   = out_channels // c_out_pmax
    n_tiles_height  = out_height // h_tile_size

    # Reshape images in X
    X_re = X.reshape((batch_size, in_channels, input_height*input_width))

    # Copy weights from HBM -> SBUF
    #weights_sbuf = nl.ndarray((out_channels, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    #nisa.dma_copy(src=W, dst=weights_sbuf)

    # Print config information
    print(" ")
    print("X dimensions: batch_size=", batch_size, "in_channels=", in_channels, "input_height=", input_height, "input_width=", input_width)
    print("W dimensions: out_channels=", out_channels, "in_channels=", in_channels, "filter_height=", filter_height, "filter_width=", filter_width)
    print("X_out dimensions: batch_size=", batch_size, "out_channels=", out_channels, "out_pool_height=", out_pool_height, "out_pool_width=", out_pool_width)

    # First iterate over every chunk of output tiles independently
    for c_out in nl.affine_range(n_tiles_c_out):
        # Copy weights for this set of output channels
        weights_sbuf = nl.ndarray((c_out_pmax, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=W[c_out*c_out_pmax:(c_out+1)*c_out_pmax,:,:,:], dst=weights_sbuf)

        # Copy bias for this set of output channels
        bias_sbuf = nl.ndarray((c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=bias[c_out*c_out_pmax:(c_out+1)*c_out_pmax,], dst=bias_sbuf)

        # Process the images in batches
        for b in nl.affine_range(batch_size):
            # Height tiles
            for h in nl.affine_range(n_tiles_height):
                # Allocate result tensor in PSUM
                res_psum = nl.zeros((c_out_pmax, h_tile_size * out_width), dtype=nl.float32, buffer=nl.psum)

                for c in nl.sequential_range(n_tiles_c_in):
                    # Copy image from HBM -> SBUF
                    image_sbuf = nl.ndarray((c_in_pmax, (h_tile_size+filter_height-1) * out_width), dtype=X.dtype, buffer=nl.sbuf)

                    # Iterate over every element of the filter
                    for j in nl.sequential_range(filter_width):
                        #shift the image everytime we move the width #Likely very inefficient but maybe a good starting point.
                        for r in nl.affine_range(h_tile_size+filter_height-1):
                            nisa.dma_copy(src=X_re[b,c*c_in_pmax:(c+1)*c_in_pmax,(r+h*h_tile_size)*input_width+j:(r+h*h_tile_size)*input_width+j+out_width],
                                    dst=image_sbuf[:,r*out_width:(r+1)*out_width])

                        for i in nl.affine_range(filter_height):
                            # Generate weights matrix
                            weights_ij  = weights_sbuf[:, c*c_in_pmax:(c+1)*c_in_pmax, i, j]
                            weights_ijT = nisa.nc_transpose(weights_ij)
                            weights_ijT_sbuf = nisa.tensor_copy(weights_ijT)

                            # Shift input image - not sure if this flattened filter approach works
                            image_sbuf_row_start = i*out_width
                            image_sbuf_row_end   = image_sbuf_row_start + h_tile_size * out_width
                            res_psum += nisa.nc_matmul( weights_ijT_sbuf, image_sbuf[:, image_sbuf_row_start:image_sbuf_row_end]) #input already in transposed format

                # Move result to SBUF
                res_sbuf = nisa.tensor_copy(res_psum)

                # Add bias to result
                res_sbuf = nisa.tensor_tensor(res_sbuf, bias_sbuf, nl.add)

                # Move result to HBM
                nisa.dma_copy(src=res_sbuf, dst=X_out[b, c_out*c_out_pmax:(c_out+1)*c_out_pmax,
                    h*h_tile_size*out_width:(h+1)*h_tile_size*out_width])

    return X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
