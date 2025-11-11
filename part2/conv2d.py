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
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Reshape images in X - does this work right?
    X_re = X.reshape((batch_size, in_channels, input_height*input_width))

    # Copy weights from HBM -> SBUF
    weights_sbuf = nl.ndarray((out_channels, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=W, dst=weights_sbuf)

    # Print config information
    print(" ")
    print("X dimensions: batch_size=", batch_size, "in_channels=", in_channels, "input_height=", input_height, "input_width=", input_width)
    print("W dimensions: out_channels=", out_channels, "in_channels=", in_channels, "filter_height=", filter_height, "filter_width=", filter_width)
    print("X_out dimensions: batch_size=", batch_size, "out_channels=", out_channels, "out_pool_height=", out_pool_height, "out_pool_width=", out_pool_width)

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # Copy image from HBM -> SBUF
        image_sbuf = nl.ndarray((in_channels, input_height * input_width), dtype=X.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=X_re[b], dst=image_sbuf)

        # Allocate result tensor in PSUM
        res_psum = nl.zeros((out_channels, out_height * out_width), dtype=X.dtype, buffer=nl.psum)

        # Iterate over every element of the filter
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                # Generate weights matrix
                # this is an attempt to pick out one set of elements from each filter, and then transpose to get
                # the right dimensions for nc_matmul
                weights_ij  = weights_sbuf[:,:,i,j]
                weights_ijT = nisa.nc_transpose(weights_ij)
                weights_ijT_sbuf = nisa.tensor_copy(weights_ijT)

                # Shift input image - not sure if this flattened filter approach works
                image_sbuf_row_start = i*filter_width + j
                image_sbuf_row_end   = image_sbuf_row_start + out_height * out_width
                res_psum += nisa.nc_matmul(weights_ijT_sbuf, image_sbuf[:, image_sbuf_row_start:image_sbuf_row_end]) #changed as the left hand side should be transposed for performance i think?

        # Move result to SBUF
        res_sbuf = nisa.tensor_copy(res_psum)

        # Transpose result to get [out_channels, out_height * out_width]
        # res_psum_T = nisa.nc_transpose(res_sbuf)

        # Move result to SBUF
        # res_sbuf_T = nisa.tensor_copy(res_psum_T)

        # Move result to HBM
        # X_out_temp = nl.ndarray((out_channels, out_height * out_width), dtype=X.dtype, buffer=nl.hbm)
        nisa.dma_copy(src=res_sbuf, dst=X_out[b,:,:]) #do the reshape at the absolute end

        # Reshape result to match expected dimensions
        # This portion does not work
        # X_out[b,:,:,:] = X_out_temp.reshape((out_channels, out_pool_height, out_pool_width))

        #raise RuntimeError("Please fill your implementation of computing convolution"
        #                   " of X[b] with the weights W and bias b, followed by a"
        #                   " maxpool and store the result in X_out[b]")

    return X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
