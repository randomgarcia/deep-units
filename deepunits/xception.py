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

class XceptionBase(DeepUnit):
    def __init__(
        self,
        features,
        depth_multiplier=1,
        activation='relu',
        inter_activation=False,
        batch_norm=True,
    ):
        units = OrderedDict()
        
        if inter_activation:
            use_act = activation
        else:
            use_act = None
        
        if type(features) not in [list,tuple]:
            features = [features]
        
        for ii in range(len(features)):
            units[f'sep_conv{ii}'] = SeparableConv2D(
                features[ii],
                (3, 3),
                padding='same',
                use_bias=False,
                depth_multiplier=depth_multiplier,
                activation=use_act,
            )
            if batch_norm:
                units[f'batch_norm{ii}'] = BatchNormalization(axis=-1)

            if not inter_activation:
                if type(activation) is str:
                    units[f'activation{ii}'] = Activation(activation)
                else:
                    units[f'Activation{ii}'] = activation
        
        super().__init__(units)
    
    @classmethod
    def create(
        cls,
        features,
        repeats=2,
        depth_multiplier=1,
        activation='relu',
        inter_activation=False,
        batch_norm=True,
    ):
        """
        All scalar alternative constructor for use in model creation
        """
        return cls(
            repeats*[features],
            depth_multiplier=1,
            activation='relu',
            inter_activation=False,
            batch_norm=True,
        )
        

class XceptionUnit(ResidualUnit):
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
        residual_branch=None,
    ):
        main_branch = XceptionBase(
            features,
            depth_multiplier=depth_multiplier,
            activation=activation,
            inter_activation=inter_activation,
            batch_norm=batch_norm,
        )
        
        super().__init__(main_branch,residual_branch=residual_branch)
        
    @classmethod
    def create(
        cls,
        features,
        repeats=2,
        depth_multiplier=1,
        activation='relu',
        inter_activation=False,
        batch_norm=True,
    ):
        """
        All scalar alternative constructor for use in model creation
        """
        return cls(
            repeats*[features],
            depth_multiplier=1,
            activation='relu',
            inter_activation=False,
            batch_norm=True,
        )
        
        