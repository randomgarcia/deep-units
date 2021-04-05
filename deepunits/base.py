import tensorflow as tf 

from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    Activation,
)

class DeepUnit:
    def __init__(self):
        
        # could use an OrderedDict
        self.Units = {}
        self.Tensors = {}
        self.Names = []
    
    def __call__(self,x):
        
        z = x 
        for currname in self.Names:
            z = self.Units[currname](z)
            self.Tensors[currname] = z
        
        return z 
        