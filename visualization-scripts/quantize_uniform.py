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
