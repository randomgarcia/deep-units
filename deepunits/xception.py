from .base import DeepUnit, ResidualUnit

from collections import OrderedDict
from tensorflow.keras.layers import (
    DepthwiseConv2D, 
    SeparableConv2D,
    Conv2D,
    BatchNormalization,
    Activation,
)

# DepthwiseConv2D is the stacked filter part only
# SeparableConv2D is the stacked filter followed by a 1x1 convolution
# therefore a modification of the Xception block is a 1x1 conv followed by a depthwise conv2D

class XceptionUnit(DeepUnit):
    def __init__(
        self,
        features,
        depth_multiplier=1,
        activation='relu',
        batch_norm=True,
        activation_at_start=True,
        activation_at_end=False,
        maxpool_at_end=True,
        postproc=None,
        force_residual_conv=False,
        preproc=None,
    ):
        units = OrderedDict()
        
        if isinstance(maxpool_at_end,bool) and maxpool_at_end:
            maxpool_at_end = 'm2'
        
        if type(features) not in [list,tuple]:
            features = [features]
        
        for ii in range(len(features)):
            if (ii>0) or activation_at_start:
                if type(activation) is str:
                    activ = self.validate_activation(activation)
                    units[f'activation{ii}'] = activ
                else:
                    units[f'activation{ii}'] = activation

            units[f'sep_conv{ii}'] = SeparableConv2D(
                features[ii],
                (3, 3),
                padding='same',
                use_bias=not batch_norm,
                depth_multiplier=depth_multiplier,
            )
            
            if batch_norm:
                units[f'batch_norm{ii}'] = BatchNormalization(axis=-1)
        
        if activation_at_end:
            if type(activation) is str:
                units['activation{0}'.format(len(features)+1)] = Activation(activation)
            else:
                units['activation{0}'.format(len(features)+1)] = activation
        
        if maxpool_at_end:
            postproc = 'm2'
        
            
        super().__init__(units,preproc=preproc,postproc=postproc)
    
    @classmethod
    def create(
        cls,
        features,
        repeats=2,
        depth_multiplier=1,
        activation='relu',
        batch_norm=True,
        activation_at_start=True,
        activation_at_end=False,
        maxpool_at_end=True,
        postproc=None,
    ):
        """
        All scalar alternative constructor for use in model creation
        """
        return cls(
            repeats*[features],
            depth_multiplier=depth_multiplier,
            activation=activation,
            batch_norm=batch_norm,
            activation_at_start=activation_at_start,
            activation_at_end=activation_at_end,
            maxpool_at_end=maxpool_at_end,
            postproc=postproc,
        )


class XceptionResidualUnit(ResidualUnit):
    """
    This one does the spatial conv first followed by the depthwise conv
    
    Have an intermediate activation and a final activation
    """
    def __init__(
        self,
        features,
        depth_multiplier=1,
        activation='relu',
        inter_activation=False,
        batch_norm=True,
        activation_at_start=True,
        activation_at_end=False,
        maxpool_at_end=True,
        preproc=None,
        postproc=None,
        force_residual_conv=False,
    ):
        
        main_branch = XceptionUnit(
            features,
            depth_multiplier=depth_multiplier,
            activation=activation,
            batch_norm=batch_norm,
            activation_at_start=activation_at_start,
            activation_at_end=activation_at_end,
            maxpool_at_end=maxpool_at_end,
            preproc=preproc,
            postproc=postproc,
        )

        if (isinstance(preproc,str)) and (preproc.lower().startswith('u')):
            resunits = OrderedDict()
            resunits['feature'] = Conv2D(
                features[-1]*depth_multiplier,
                (1,1),
                strides=(1,1),
                padding='same',
                use_bias=not batch_norm,
            )
            if batch_norm:
                resunits['batch_norm'] = BatchNormalization(axis=-1)
            
            # resunits['upscale'] = UpSampling2D((2,2))
            resunits['upscale'] = self.validate_layer(preproc)
            residual_branch = DeepUnit(resunits)

        elif maxpool_at_end or postproc or force_residual_conv:
            
                if maxpool_at_end or postproc:
                    strides = (2,2)
                else:
                    strides = (1,1)
                
                resunits = OrderedDict()
                resunits['feature'] = Conv2D(
                    features[-1]*depth_multiplier,
                    (1,1),
                    strides=strides,
                    padding='same',
                    use_bias=not batch_norm,
                )
                if batch_norm:
                    resunits['batch_norm'] = BatchNormalization(axis=-1)
                residual_branch = DeepUnit(resunits)
        else:
            residual_branch = None
        
        super().__init__(main_branch,residual_branch=residual_branch)
        
    @classmethod
    def create(
        cls,
        features,
        repeats=2,
        depth_multiplier=1,
        activation='relu',
        batch_norm=True,
        activation_at_start=True,
        activation_at_end=False,
        maxpool_at_end=True,
        postproc=None,
    ):
        """
        All scalar alternative constructor for use in model creation
        """
        return cls(
            repeats*[features],
            depth_multiplier=1,
            activation=activation,
            batch_norm=batch_norm,
            activation_at_start=activation_at_start,
            activation_at_end=activation_at_end,
            maxpool_at_end=maxpool_at_end,
            postproc=postproc,
        )
        
def create_xception_units(
    features,
    repeats=2,
    residual=True,
    num_units=4,
    depth_multiplier=1,
    activation='relu',
    batch_norm=True,
    activation_at_start=True,
    activation_at_end=False,
    maxpool_at_end=False,
    maxpool_after_first=True,
    postproc=None,
    force_residual_conv=False,
):
    """
    This creates one scale of an xception model (ie, no intermediate downscaling)
    
    Usually it will be either num_units=1 and maxpool_at_end=True
    or num_units>0 and maxpool_at_end=False
    """
    
    if isinstance(maxpool_at_end,bool) and maxpool_at_end:
        maxpool_at_end = 'm2'
    if isinstance(maxpool_after_first,bool) and maxpool_after_first:
        maxpool_after_first = 'm2'
    

    units = OrderedDict()
    
    if residual:
        constructor = XceptionResidualUnit
    else:
        constructor = XceptionUnit
        
    for ii in range(num_units):
        if ((ii==0) and (maxpool_after_first)) or ((ii==(num_units-1)) and (maxpool_at_end)):
            do_maxpool = True
        else:
            do_maxpool = False

        # think we can work out whether a res_conv is needed based on whether maxpooling is performed
        if (ii==0) and force_residual_conv:
            res_conv = True
        else:
            res_conv = False

        units[f'Xc{ii}'] = constructor(
            repeats*[features],
            depth_multiplier=1,
            activation=activation,
            batch_norm=batch_norm,
            activation_at_start=activation_at_start,
            activation_at_end=activation_at_end,
            maxpool_at_end=do_maxpool,
            force_residual_conv=res_conv,
        )
    
    return DeepUnit(units)
    
def upscaling_xception_units(
    features,
    repeats=2,
    residual=True,
    num_units=4,
    depth_multiplier=1,
    activation='relu',
    batch_norm=True,
    activation_at_start=True,
    activation_at_end=False,
    force_residual_conv=False,
):
    """
    This creates one scale of an xception model (ie, no intermediate downscaling)
    
    Usually it will be either num_units=1 and maxpool_at_end=True
    or num_units>0 and maxpool_at_end=False
    """
    
    units = OrderedDict()
    
    if residual:
        constructor = XceptionResidualUnit
    else:
        constructor = XceptionUnit
        
    for ii in range(num_units):
        if (ii==0):
            preproc='u2'
        else:
            preproc = None

        # think we can work out whether a res_conv is needed based on whether maxpooling is performed
        if (ii==0):
            res_conv = True
        else:
            res_conv = False

        units[f'Xc{ii}'] = constructor(
            repeats*[features],
            depth_multiplier=1,
            activation=activation,
            batch_norm=batch_norm,
            activation_at_start=activation_at_start,
            activation_at_end=activation_at_end,
            maxpool_at_end=False,
            force_residual_conv=res_conv,
            preproc=preproc,
        )
    
    return DeepUnit(units)
    