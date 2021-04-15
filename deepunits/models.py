from .base import DeepUnit, ConvUnit, FCUnit
from .densenet import StandardDenseNetUnit
from collections import OrderedDict


class DeepModel(DeepUnit):
    def __init__(self,base_unit,base_kws,fc_kws,preproc=24,postproc=None,fcunit=None):
        
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
                    base_kws[ii]['postproc'] = '-1'
                else:
                    base_kws[ii]['postproc'] = '2'
        
        units = OrderedDict()
        for ii in range(len(base_kws)):
            name = 'Unit{0}'.format(ii)
            units[name] = base_unit(**base_kws[ii])
        
        units['FCUnit'] = fcunit(**fc_kws)
        
        super().__init__(units,names=None,preproc=preproc,postproc=postproc)
    
    @classmethod
    def dense_net(cls,preproc=24,growth_rate=[4,8,12],repeats=[2,4,6],output_features=[12,24,48],fcfeat=[36,10],**kwargs):
        """
        Convert to kws for the constructor
        """
        kws = [
            {
                'growth_rate':growth_rate[ii],
                'repeats':repeats[ii],
                'output_features':output_features[ii],
            }
            for ii in range(len(growth_rate))
        ]
        
        fc_kws = {'dense_features':fcfeat}
        
        return cls(StandardDenseNetUnit,base_kws=kws,fc_kws=fc_kws,preproc=preproc,**kwargs)
        