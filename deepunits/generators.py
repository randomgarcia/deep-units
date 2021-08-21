import numpy as np
import pandas as pd
import tensorflow as tf

__all__ = [
    'join_batches',
    'get_batches',
]

def join_batches(x):
    if type(x[0]) is list:
        return [join_batches(xx) for xx in list(zip(*x))]
    elif type(x[0]) is tuple:
        return tuple([join_batches(xx) for xx in list(zip(*x))])
    elif isinstance(x[0],np.ndarray):
        return np.concatenate(x,0)
    elif isinstance(x[0],pd.DataFrame):
        return pd.concat(x,0,ignore_index=True,sort=False)
    elif isinstance(x[0],tf.Tensor):
        raise NotImplementedError
    else:
        raise TypeError("File type not handled yet")

def get_batches(gen,n=20):
    xy = [gen.__next__() for ii in range(n)]
    
    return join_batches(xy)
