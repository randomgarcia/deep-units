from .base import (
    DeepUnit, 
    ConvUnit, 
    to_ordered_dict, 
    to_listnames,
    TensorDict,
)
from tensorflow.keras.layers import (
    LeakyReLU, 
    Concatenate,
)

class DenseNetUnit(DeepUnit):
    """
    Class containing the basic dense net unit architecture, ie concatenating after each
    unit
    """
    def __init__(self,units,names=None,preproc=None,postproc=None,final_layer=None):
        
        # don't need the structure to follow exactly at the moment
        # also need the concatenation parts, so break back into list
        if names is None:
            names = ['DenseNet{0}'.format(ii) for ii in range(len(units))]
        units,names = to_listnames(units,names)
        
        # interleave with concatenation layers
        concatunits = [Concatenate(axis=-1) for x in range(len(names))]
        
        self.Units = {}
        self.Units['Features'] = to_ordered_dict(units,names)
        self.Units['Concats'] = to_ordered_dict(concatunits,names)
        
        self.Tensors = {}
        self.Tensors['Features'] = TensorDict()
        self.Tensors['Concats'] = TensorDict()
        
        
        if final_layer is not None:
            self.Units['FinalLayer'] = self.validate_layer(final_layer)
            self.Tensors['Other'] = TensorDict()
            
        self.set_pre_post_proc(preproc,postproc)
        
    def tf_call(self,x):
        """override the standard call to build the densenet unit"""
        z = x
        for currname in self.Units['Features'].keys():
            z1 = self.Units['Features'][currname](z)
            self.Tensors['Features'][currname] = z1
            
            z = self.Units['Concats'][currname]([z,z1])
            self.Tensors['Concats'][currname] = z
        
        if 'FinalLayer' in self.Units.keys():
            z = self.Units['FinalLayer'](z)
            self.Tensors['Other']['FinalLayer'] = z

        return z
    

    
class StandardDenseNetUnit(DenseNetUnit):
    """
    The basic DenseNet unit consists of a 1x1 conv followed by a 3x3 conv, concatenating the result
    and repeating
    """
    def __init__(
        self,
        growth_rate,
        repeats,
        output_features=None,
        bottleneck=None,
        preproc=None,
        postproc=None,
        activation='leakyrelu',
    ):
        
        if type(growth_rate) not in [list,tuple]:
            growth_rate = repeats*[growth_rate]
        
        growth_rate = [int(x) for x in growth_rate]
        repeats = len(growth_rate)
        
        if bottleneck is None:
            bottleneck = [4*x for x in growth_rate]
        elif type(bottleneck) not in [list,tuple]:
            bottleneck = repeats*[bottleneck]
        
        if (output_features is not None) and (output_features<0):
            output_features = int(0.5 * sum(growth_rate))
        
        
        units = []
        for rr in range(repeats):
            if activation=='leakyrelu':
                acts = [LeakyReLU(alpha=0.1),LeakyReLU(alpha=0.1)]
            else:
                acts = activation
                
            unit0 = ConvUnit(
                features=[bottleneck[rr],growth_rate[rr]],
                kernel_sizes=[1,3],
                activations=acts,
            )
            
            units.append(unit0)
        
        # sort out the names later
        super().__init__(units,preproc=preproc,postproc=postproc,final_layer=output_features)
        