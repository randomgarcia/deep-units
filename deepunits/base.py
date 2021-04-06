import tensorflow as tf 

from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    Activation,
)

from collections import OrderedDict

class TensorDict:
    """
    Wrapper around an OrderedDict to make accessing saved tensors easier
    """
    def __init__(self):
        self.NestedDict = OrderedDict()
    
    def __getattr__(self,*args):
        pass
    

class DeepUnit:
    def __init__(self,units=None,names=None):
        
        # could use an OrderedDict?
        self.Units = {}
        self.Tensors = {}
        self.Names = []
    
    def __call__(self,x):
        # do some checking here at some point 
        return self.tf_call(x)
    
    def tf_call(self,x,replace=True):
        
        self.Tensors['input'] = x
        z = x 
        for currname in self.Names:
            z = self.Units[currname](z)
            self.Tensors[currname] = z
        
        return z 
        