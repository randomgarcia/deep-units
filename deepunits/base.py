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
        self.CurrentKey = None
    
    def set_current_key(self,key=-1):
        self.CurrentKey = key
        return self
    
    def __getitem__(self,args):
        
        if type(args) not in [list,tuple]:
            args = (args,)
            
        if len(args)==1:
            key = self.CurrentKey if self.CurrentKey is not None else -1
            
            # need to work out which level we're accessing
            if (args[0] in self.NestedDict.keys()) or (type(args[0]) is int):
                return self.NestedDict[args[0]]
            elif args[0] in self.NestedDict[key].keys():
                return self.NestedDict[key][args[0]]
            else:
                raise ValueError
        else:
            return self.NestedDict[args[0]][args[1]]
    
    def __setitem__(self,args,val):
        if type(args) not in [list,tuple]:
            args = (args,)
            
        if len(args)==1:
            key = self.CurrentKey if self.CurrentKey is not None else -1
            
            # need to work out which level we're accessing
            if (args[0] in self.NestedDict.keys()) or (type(args[0]) is int):
                # self.NestedDict[args[0]] = val
                raise ValueError("Can't set entire dict")
            # elif args[0] in self.NestedDict[key].keys():
            else:
                self.NestedDict[key][args[0]] = val 
            # else:
            #     raise ValueError
        else:
            if args[0] not in self.NestedDict.keys():
                # need to create the new dict
                self.NestedDict[args[0]] = {}
                
            self.NestedDict[args[0]][args[1]] = val
    
                
    

class DeepUnit:
    def __init__(self,units,names=None):
        
        # could use an OrderedDict?
        if type(units) is not dict:
            if type(units) not in [list,tuple]:
                units = [units]
                
            self.Units = OrderedDict()
            
            if names is None:
                names = ['Unit{0}'.format(ii) for ii in range(len(units))]
            
            for ii in range(len(units)):
                self.Units[names[ii]] = units[ii]
            
        self.Tensors = TensorDict()
        
        
    def __call__(self,x):
        # do some checking here at some point 
        return self.tf_call(x)
    
    def tf_call(self,x,replace=True):
        self.Tensors.set_current_key(x)
        # self.Tensors['input'] = x
        z = x 
        for currname in self.Units.keys():
            z = self.Units[currname](z)
            self.Tensors[currname] = z
        
        return z 
        