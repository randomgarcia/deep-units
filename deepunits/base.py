import tensorflow as tf
import numpy as np
from copy import deepcopy

from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Add,
    UpSampling2D,
)

from collections import OrderedDict

def to_ordered_dict(units,names=None):
    """
    Convert the list of units to an ordereddict
    """
    if isinstance(units,OrderedDict):
        return units
    
    if type(units) not in [list,tuple]:
        units = [units]

    unitdict = OrderedDict()

    if names is None:
        names = ['Unit{0}'.format(ii) for ii in range(len(units))]

    for ii in range(len(units)):
        unitdict[names[ii]] = units[ii]
        
    return unitdict

def to_listnames(unitdict,names=None):
    if isinstance(unitdict,OrderedDict):
        names,units = zip(*list(unitdict.items()))
    else:
        if type(unitdict) not in [list,tuple]:
            unitdict = [unitdict]
        units = unitdict
        if names is None:
            names = ['Unit{0}'.format(ii) for ii in range(len(units))]
    
    return units,names

class TensorDict:
    """
    Wrapper around an OrderedDict to make accessing saved tensors easier
    """
    def __init__(self):
        self.NestedDict = OrderedDict()
        self.CurrentKey = None

    def set_current_key(self,key=-1):
        # if isinstance(key,tf.Tensor):
        if tf.is_tensor(key):
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



# comment out the preproc and postproc stuff, this could be useful in a wrapper
class DeepUnit:
    def __init__(
        self,
        units,
        names=None,
        preproc=None,
        postproc=None,
        output_features=None,
    ):

        # could use an OrderedDict?
        self.Units = to_ordered_dict(units,names)

        self.Tensors = TensorDict()
        
        # self.set_pre_post_proc(preproc,postproc)
    
    # def set_pre_post_proc(self,preproc=None,postproc=None):
    #     self.PreProc = self.validate_layer(preproc)
    #     self.PostProc = self.validate_layer(postproc)


    def __call__(self,x):
        # do some checking here at some point
        if isinstance(self.Tensors,TensorDict):
            self.Tensors.set_current_key(x)
        else:
            for kk in self.Tensors.keys():
                self.Tensors[kk].set_current_key(x)
        
        # x = self.run_pre_processing(x)
        
        y = self.tf_call(x)
        
        # y = self.run_post_processing(y)
        
        return y

    def tf_call(self,x):
        
        z = x
        for currname in self.Units.keys():
            z = self.Units[currname](z)
            self.Tensors[currname] = z

        return z
    
    # def run_pre_processing(self,x):
    #     z = x
    #     if self.PreProc is not None:
    #         z = self.PreProc(z)
    #         self.Tensors['PreProc'] = z
        
    #     return z
    
    # def run_post_processing(self,x):
    #     z = x
    #     if self.PostProc is not None:
    #         z = self.PostProc(z)
    #         self.Tensors['PostProc'] = z
        
    #     return z
    
    
    @classmethod
    def validate_layer(cls,layer):
        """
        TODO - allow a list to specify a sequence of units
        """
        
        if isinstance(layer,int) or isinstance(layer,float):
            layer = Conv2D(layer,kernel_size=3,activation='relu',padding='same')
        
        elif isinstance(layer,str):
            # check for particular sequence to specify maxpool, conv, avpool etc
            if layer[:1].isalpha():
                # try to use a regex to parse?
                codes = layer.split('_')
                units = []
                for code0 in codes:
                    layer_type = code0[0]
                    args = [int(x) if x.isdigit() else x for x in code0[1:].split('/')]
                    
                    if layer_type.lower()=='c':
                        kwlist = ['filters','kernel_size','strides','activation']
                        kwdef = [24,3,1,'relu']
                        fixkw = {'padding':'same'}
                        ltype = Conv2D
                    elif layer_type.lower()=='m':
                        kwlist = ['pool_size','strides']
                        kwdef = [2,2]
                        fixkw = {}
                        ltype = MaxPooling2D
                    elif layer_type.lower()=='a':
                        kwlist = ['pool_size','strides']
                        kwdef = [2,2]
                        fixkw = {}
                        ltype = AveragePooling2D 
                    elif layer_type.lower()=='u':
                        kwlist = ['size','interpolation']
                        kwdef = [2,'nearest']
                        fixkw = {}
                        ltype = UpSampling2D
                    elif layer_type.lower()=='g':
                        ltype = GlobalAveragePooling2D
                        kwlist = []
                        kwdef = []
                        fixkw = {}
                        
                    
                    usev = kwdef.copy()
                    usev[:len(args)] = [x if x!='' else y for x,y in zip(args,usev)]
                    
                    kws = {**fixkw, **{k:v for k,v in zip(kwlist,usev)}}
                    if ('activation' in kws.keys()) and (isinstance(kws['activation'],str)):
                        kws['activation'] = Activation(kws['activation'])
                    
                    layer0 = ltype(**kws)
                    units.append(layer0)
                layer = DeepUnit(units)
                    
            else:
                factor = int(layer)
                if factor<=0:
                    layer = GlobalAveragePooling2D()
                else:
                    layer = MaxPooling2D(factor)
                
        return layer
    
    @classmethod
    def validate_activation(cls,layer):
        if isinstance(layer,str):
            if layer in ['relu','softmax','sigmoid','tanh']:
                return Activation(layer)
            
            layer_type = layer[:1]
            args = [int(x) if x.isdigit() else x for x in layer[1:].split('/')]
            if layer_type.lower() in ['c','m']:
                kwlist = []
                kwdef = []
                fixkw = {'activation':'softmax'}
                ltype = Activation
            elif layer_type.lower()=='r':
                kwlist = []
                kwdef = []
                fixkw = {'activation':'relu'}
                ltype = Activation
            elif layer_type.lower()=='t':
                kwlist = []
                kwdef = []
                fixkw = {'activation':'tanh'}
                ltype = Activation
            elif layer_type.lower()=='s':
                kwlist = []
                kwdef = []
                fixkw = {'activation':'sigmoid'}
                ltype = Activation
            elif layer_type.lower()=='l':
                kwlist = ['alpha']
                kwdef = [0.2]
                fixkw = {}
                ltype = tf.keras.layers.LeakyReLU
            
            
            usev = kwdef.copy()
            usev[:len(args)] = [x if x!='' else y for x,y in zip(args,usev)]
            
            kws = {**fixkw, **{k:v for k,v in zip(kwlist,usev)}}
            
            return ltype(**kws)
                    
        else:
            return layer
        

class BlockWrapper(DeepUnit):
    """
    A wrapper around a basic unit adding maxpooling, residual connection, etc
    """
    def __init__(self,unit,initial_maxpool=False,final_maxpool=True,residual=False,residual_feat=None):
        
        unitdict = {}
        unitdict['BaseUnit'] = unit
        
        if initial_maxpool:
            unitdict['InitialOperation'] = validate_layer('m2')
        else:
            unitdict['InitialOperation'] = NullUnit()
        
        if final_maxpool:
            unitdict['FinalOperation'] = validate_layer('m2')
        else:
            unitdict['FinalOperation'] = NullUnit()
        
        self.Residual = residual
        if residual:
            stride = 1 
            if initial_maxpool:
                stride = stride * 2
            if final_maxpool:
                stride = stride * 2 
            
            
            if residual_feat is None:
                # see if we can get it from the DeepUnit
                residual_feat = unit.OutputFeatures
            
            if (residual_feat is not None):
                unitdict['ResidualBranch'] = Conv2D(residual_feat,1,strides=stride,padding='same')
            else:
                unitdict['ResidualBranch'] = NullUnit()
                
            unitdict['Combine'] = Add()
        
        super().__init__(unitdict)
    
    def tf_call(self,X):
        Z = self.Units['InitialOperation'](X)
        self.Tensors['InitialOperation'] = Z 
        
        Z = self.Units['BaseUnit'](Z)
        self.Tensors['BaseUnit'] = Z 
        
        Z = self.Units['FinalOperation'](Z)
        self.Tensors['FinalOperation'] = Z
        
        if self.Residual:
            R = self.Units['ResidualBranch'](X)
            self.Tensors['ResidualBranch'] = R
            
            Z = self.Units['Combine']([Z,R])
            self.Tensors['Combine'] = Z 
        
        return Z 
        
            
        
        
        

class NullUnit(DeepUnit):
    def __init__(self):
        super().__init__([])
    
    def tf_call(self,x):
        self.Tensors['x'] = x
        
        return x
        
class ConvUnit(DeepUnit):
    def __init__(
        self,
        features,
        kernel_sizes=3,
        strides=1,
        activations='relu',
        batch_norm=False,
        padding='same',
        preproc=None,
        postproc=None,
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

        super().__init__(units,names,preproc=preproc,postproc=postproc)


class BACUnit(DeepUnit):
    """
    Batch norm, Activation, Convolution
    """
    def __init__(
        self,
        features,
        kernel_sizes=3,
        strides=1,
        activations='relu',
        batch_norm=False,
        padding='same',
        preproc=None,
        postproc=None,
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
            activations = [deepcopy(activations)]*len(features)
        if type(padding) not in [list,tuple,np.ndarray]:
            padding = [padding]*len(features)


        activations = [
            Activation(aa) if type(aa) is str else aa
            for aa in activations
        ]

        convlayers = [
            Conv2D(ff,kernel_size=kk,strides=ss,padding=pp)
            for ff,kk,ss,pp in zip(features,kernel_sizes,strides,padding)
        ]


        names = []
        units = []
        for ii in range(len(features)):
            if batch_norm:
                names.append('BatchNorm_{0}'.format(ii))
                units.append(BatchNormalization())
            
            if activations[ii] is not None:
                names.append('Activation_{0}'.format(ii))
                units.append(activations[ii])
            
            names.append('Conv2D_{0}'.format(ii))
            units.append(convlayers[ii])
            
            
            
        super().__init__(units,names,preproc=preproc,postproc=postproc)


class FCUnit(DeepUnit):
    def __init__(self,dense_features,activations='relu',last_activation='softmax'):
        
        if type(dense_features) not in [list,tuple,np.ndarray]:
            dense_features = [dense_features]

        if type(activations) not in [list,tuple,np.ndarray]:
            activations = [activations]*len(dense_features)
        
        if len(activations)==len(dense_features):
            activations = activations[:-1]
        
        activations = activations + [last_activation]
        
        
        denselayers = [
            Dense(ff)
            for ff in dense_features
        ]

        activations = [
            Activation(aa) if type(aa) is str else aa
            for aa in activations
        ]

        names = []
        units = []
        for ii in range(len(dense_features)):
            names.append('Dense_{0}'.format(ii))
            units.append(denselayers[ii])

            if activations[ii] is not None:
                names.append('Activation_{0}'.format(ii))
                units.append(activations[ii])

        super().__init__(units,names,preproc=None,postproc=None)


class ResidualUnit(DeepUnit):
    def __init__(self,main_branch,residual_branch=None):
        
        if residual_branch is None:
            residual_branch = NullUnit()
        
        units = OrderedDict()
        units['main_branch'] = main_branch
        units['residual_branch'] = residual_branch
        
        units['merge'] = Add()
        
        super().__init__(units)
    
    def tf_call(self,x):
        z1 = self.Units['main_branch'](x)
        self.Tensors['main_branch'] = z1
        
        z2 = self.Units['residual_branch'](x)
        self.Tensors['residual_branch'] = z2
        
        y = self.Units['merge']([z1,z2])
        self.Tensors['merge'] = y
        
        return y
        
        
        