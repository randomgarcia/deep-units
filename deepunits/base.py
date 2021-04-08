import tensorflow as tf
import numpy as np

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
        if isinstance(key,tf.Tensor):
            key = key.ref()
        self.CurrentKey = key

        if key not in self.NestedDict.keys():
            self.NestedDict[key] = {}

        return self

    def __getitem__(self,args):

        if type(args) not in [list,tuple]:
            args = (args,)

        if len(args)==1:
            key = self.CurrentKey if self.CurrentKey is not None else -1
            if isinstance(key,tf.Tensor):
                key = key.ref()
            # need to work out which level we're accessing
            if isinstance(args[0],tf.Tensor):
                arg0 = args[0].ref()
            else:
                arg0 = args[0]

            if (arg0 in self.NestedDict.keys()) or (type(arg0) is int):
                return self.NestedDict[arg0]
            elif args[0] in self.NestedDict[key].keys():
                return self.NestedDict[key][args[0]]
            else:
                raise ValueError
        else:
            if isinstance(args[0],tf.Tensor):
                arg0 = args[0].ref()
            else:
                arg0 = args[0]

            return self.NestedDict[arg0][args[1]]

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
    def __init__(self,units,names=None,preproc=None,postproc=None):

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
        self.Tensors.set_current_key(x)
        x = self.run_pre_processing(x)
        
        y = self.tf_call(x)
        
        y = self.run_post_processing(y)
        
        return y

    def tf_call(self,x):
        
        z = x
        for currname in self.Units.keys():
            z = self.Units[currname](z)
            self.Tensors[currname] = z

        return z
    
    def run_pre_processing(self,x):
        z = x
        if self.PreProc is not None:
            z = self.PreProc(z)
            self.Tensors['PreProc'] = z
        
        return z
    
    def run_post_processing(self,x):
        z = x
        if self.PostProc is not None:
            z = self.PostProc(z)
            self.Tensors['PostProc'] = z
        
        return z
    
    

class ConvUnit(DeepUnit):
    def __init__(
        self,
        features,
        kernel_sizes=3,
        strides=1,
        activations='relu',
        batch_norm=False,
        padding='same',
    ):

        if type(features) not in [list,tuple,np.ndarray]:
            features = [features]

        features = [int(x) for x in features]

        # make sure that the rest of the parameters are the same size
        if type(kernel_sizes) not in [list,tuple,np.ndarray]:
            kernel_sizes = [kernel_sizes]*len(features)
        if type(strides) not in [list,tuple,np.ndarray]:
            strides = [strides]*len(features)
        if type(activations) not in [list,tuple,np.ndarray]:
            activations = [activations]*len(features)
        if type(padding) not in [list,tuple,np.ndarray]:
            padding = [padding]*len(features)


        convlayers = [
            Conv2D(ff,kernel_size=kk,strides=ss,padding=pp)
            for ff,kk,ss,pp in zip(features,kernel_sizes,strides,padding)
        ]

        activations = [
            Activation(aa) if type(aa) is str else aa
            for aa in activations
        ]

        names = []
        units = []
        for ii in range(len(features)):
            names.append('Conv2D_{0}'.format(ii))
            units.append(convlayers[ii])

            if activations[ii] is not None:
                names.append('Activation_{0}'.format(ii))
                units.append(activations[ii])

        super().__init__(units,names)
