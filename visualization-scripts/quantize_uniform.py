# Created by En-Yu Yang at Harvard CS 2018/12/17
# Do uniform quantization on a numpy array
# input is in the range of [-a, a]
# and input value 0 maps to output value 0 

import numpy as np

def quantize_uniform(x, num_bits=8):
    qmin = -2.**(num_bits-1)
    qmax = 2.**(num_bits-1) -1
    scale_p = x.max() / qmax 
    scale_n = x.min() / qmin

    scale = max(scale_p, scale_n)

    q_x = (x / scale).round()
     
    return q_x*scale


def dequantize_uniform(q_x, scale):
    return q_x*scale
