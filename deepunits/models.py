from .base import DeepUnit, ConvUnit, FCUnit
from .densenet import StandardDenseNetUnit
from .xception import create_xception_units
from collections import OrderedDict


class DeepModel(DeepUnit):
    def __init__(self,base_unit,base_kws,fc_kws,prefc_pooling=None,preproc=24,postproc=None,fcunit=None):
        
        # by default add maxpooling and then globalaverage pooling to
        # the base units
        
        # if make sure that each parameter is a scalar, can get the
        # number of units from the kws length
        
        if fcunit is None:
            fcunit = FCUnit
            
        # convert to list of dicts
        if type(base_kws) is list:
            # separate kws for each unit
            num_units = len(base_kws)
        elif type(base_kws) is dict:
            num_units = max([len(x) if type(x) is list else 1 for k,x in base_kws.items()])
            
            base_kws = [
                {
                    k:(v[ii] if type(v) is list else v) 
                    for k,v in base_kws.items()
                } 
                for ii in range(num_units)
            ]
        else:
            raise TypeError('base_kws needs to be list of dict or dict of lists')
        
        for ii in range(len(base_kws)):
            if 'postproc' not in base_kws[ii].keys():
                if ii==(len(base_kws)-1):
                    base_kws[ii]['postproc'] = 'g'
                else:
                    base_kws[ii]['postproc'] = 'm2'
        
        units = OrderedDict()
        for ii in range(len(base_kws)):
            name = 'Unit{0}'.format(ii)
            units[name] = base_unit(**base_kws[ii])

        if prefc_pooling is not None:
            prefc_pooling = self.validate_layer(prefc_pooling)
            units['PreFCPooling'] = prefc_pooling

        
        units['FCUnit'] = fcunit(**fc_kws)
        
        super().__init__(units,names=None,preproc=preproc,postproc=postproc)
    
    @classmethod
    def dense_net(
        cls,
        preproc=24,
        growth_rate=[4,8,12],
        repeats=[2,4,6],
        output_features=[12,24,48],
        fcfeat=[36,10],
        fc_kws=None,
        **kwargs
    ):
        """
        Convert to kws for the constructor
        """
        if fc_kws is None:
            fc_kws = {}
        
        kws = [
            {
                'growth_rate':growth_rate[ii],
                'repeats':repeats[ii],
                'output_features':output_features[ii],
            }
            for ii in range(len(growth_rate))
        ]
        
        fc_kws = {**fc_kws, 'dense_features':fcfeat}
        
        return cls(StandardDenseNetUnit,base_kws=kws,fc_kws=fc_kws,preproc=preproc,**kwargs)
        
    @classmethod
    def xception(
        cls,
        preproc=24,
        features=[24,48,48,72],
        residual=True,
        num_units=[1,1,4,1],
        maxpool_at_end=False,
        maxpool_after_first=[True,True,True,False],
        fc_feat=[36,10],
        fc_kws=None,
        # postproc=[None,None,None,None],
        **kwargs,
    ):
        if fc_kws is None:
            fc_kws = {}
        
        if type(num_units) not in [list,tuple]:
            num_units = len(features)*[num_units]
        if type(maxpool_at_end) not in [list,tuple]:
            maxpool_at_end = len(features)*[maxpool_at_end]
        if type(maxpool_after_first) not in [list,tuple]:
            maxpool_after_first = len(features)*[maxpool_after_first]
        if type(residual) not in [list,tuple]:
            residual = len(features)*[residual]
        # if type(postproc) not in [list,tuple]:
        #     postproc = len(features)*[postproc]
        
        

        kws = [
            {
                'features':features[ii],
                'residual':residual[ii],
                'num_units':num_units[ii],
                'maxpool_at_end':maxpool_at_end[ii],
                'maxpool_after_first':maxpool_after_first[ii],
                'force_residual_conv':not maxpool_after_first[ii],
                'postproc':None,
            }
            for ii in range(len(features))
        ]
        
        fc_kws = {**fc_kws, 'dense_features':fc_feat}
        
        return cls(
            create_xception_units,
            base_kws=kws,
            fc_kws=fc_kws,
            preproc=preproc,
            prefc_pooling='g',
            **kwargs
        )
        

class ResidualModel(DeepModel):
    """
    Create a model with residual connections between the downscaling operations
    """
    def __init__(self):
        """
        Need to think about how the residual connection will be handled, as it's
        likely to go in partway into the next block
        """
        pass
    