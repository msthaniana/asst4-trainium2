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
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax   = nl.tile_size.pmax
    c_out_pmax  = nl.tile_size.pmax
    h_tile_size = 2 # Operate on two rows at a time

    n_tiles_c_in    = in_channels // c_in_pmax
    n_tiles_c_out   = out_channels // c_out_pmax
    n_tiles_height  = out_height // h_tile_size

    # First iterate over every chunk of output tiles independently
    for c_out in nl.affine_range(n_tiles_c_out):
        # Copy weights for this set of output channels
        weights_sbuf = nl.ndarray((c_out_pmax, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=W[c_out*c_out_pmax:(c_out+1)*c_out_pmax,:,:,:], dst=weights_sbuf)

        #Generate a weight matrix that is transposed
        weights_ijT_sbuf = nl.ndarray((c_out_pmax, c_in_pmax, n_tiles_c_in, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        for c in nl.affine_range(n_tiles_c_in):
            for j in nl.affine_range(filter_width):
                for i in nl.affine_range(filter_height):
                    weights_ijT = nisa.nc_transpose(weights_sbuf[:, c*c_in_pmax:(c+1)*c_in_pmax, i, j])
                    weights_ijT_sbuf[:,:,c,i,j] = nisa.tensor_copy(weights_ijT)

        # Copy bias for this set of output channels
        bias_sbuf = nl.ndarray((c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=bias[c_out*c_out_pmax:(c_out+1)*c_out_pmax,], dst=bias_sbuf)

        # Process the images in batches
        for b in nl.affine_range(batch_size):
            pool_res = nl.ndarray((c_out_pmax, out_pool_height, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
            
            # Height tiles
            for h in nl.affine_range(n_tiles_height):
                # Allocate result tensor in PSUM
                res_psum = nl.zeros((c_out_pmax, h_tile_size, out_width), dtype=nl.float32, buffer=nl.psum)

                for c in nl.affine_range(n_tiles_c_in):
                    # Copy image from HBM -> SBUF
                    image_sbuf = nl.ndarray((c_in_pmax, (h_tile_size+filter_height-1), input_width), dtype=X.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=X[b,c*c_in_pmax:(c+1)*c_in_pmax,h*h_tile_size:h_tile_size+filter_height-1+h*h_tile_size,:], dst=image_sbuf)

                    # Iterate over every element of the filter
                    for j in nl.sequential_range(filter_width):
                        for i in nl.affine_range(filter_height):
                            res_psum += nisa.nc_matmul(weights_ijT_sbuf[:,:,c,i,j], image_sbuf[:,i:i+h_tile_size,j:j+out_width])

                # Add bias to result
                res_sbuf = nisa.tensor_tensor(res_psum, bias_sbuf, nl.add)

                if (pool_size == 1):
                    # Reshape to apply max_pool locality
                    res_sbuf_re = res_sbuf.reshape((c_out_pmax, h_tile_size, out_width))
                    pool_res[:,h*h_tile_size:h*h_tile_size+h_tile_size,] = res_sbuf_re
                else: #should work for any non-one pool size
                    # Reshape to apply max_pool locality
                    res_sbuf_re2 = res_sbuf.reshape((c_out_pmax, h_tile_size/pool_size, pool_size, out_width/pool_size, pool_size))
                    pool_res[:,h:h+1,:] = nisa.tensor_reduce(nl.max, res_sbuf_re2, axis=[2,4])

            nisa.dma_copy(src=pool_res, dst=X_out[b, c_out*c_out_pmax:(c_out+1)*c_out_pmax,:, :])

    return X_out
