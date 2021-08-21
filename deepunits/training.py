import tensorflow as tf 
from collections import defaultdict

class ModelOutput:
    def __init__(self,name=None):
        self.Name = name
    def metrics(self):
        raise NotImplementedError
    def metric_dict(self):
        metrics = self.metrics()
        if metrics is None:
            return {}
        else:
            return {self.get_name():self.metrics()}
            
    def get_metrics(self):
        """
        If a name has been supplied, return the dict, otherwise return a list
        
        How does this work for mix of named and unnamed outputs?
        """
        name = self.get_name()
        
        if name is None:
            return self.metrics()
        else:
            return self.metric_dict()
            
    def loss(self):
        raise NotImplementedError
    def get_name(self):
        return self.Name
    
    @classmethod
    def from_model(cls,mdl,types):
        """
        can we get the layer output names directly from the model?
        """
        raise NotImplementedError("ToDo")

class ClassificationOutput(ModelOutput):
    def __init__(self,name=None):
        super().__init__(name)
    def metrics(self):
        return ['accuracy']
    def loss(self):
        return 'categorical_crossentropy'
        
class BinaryOutput(ModelOutput):
    def __init__(self,name=None):
        super().__init__(name)
    def metrics(self):
        return ['accuracy']
    def loss(self):
        return 'binary_crossentropy'
        
class RegressionOutput(ModelOutput):
    def __init__(self,name=None):
        super().__init__(name)
    def metrics(self):
        return []
    def loss(self):
        return 'mse'
        
class CosineOutput(ModelOutput):
    def __init__(self,name=None):
        super().__init__(name)
    def metrics(self):
        return []
    def loss(self):
        raise NotImplementedError
    
        

    
        

class TrainingRun:
    def __init__(self,model,strategy=None):
        self.Model = model
        
        self.Strategy = strategy
        
        self.Callbacks = []
        self.OutputFolder = None
    
    def set_output_folder(self,folder):
        self.OutputFolder = folder
        
        return self
    
    def set_optimizer_fun(self,optfun):
        """
        The only parameter to optfun should be the learning rate - use a partial
        to provide other arguments
        """
        self.OptimizerFunction = optfun
        
        return self
    
    def set_outputs(self,model_outputs,loss_weights=None):
        """
        At some point may do this automatically, but not yet, not yet..
        """
        self.ModelOutputs = model_outputs
        self.LossWeights = loss_weights
        
        return self
    
    def get_losses(self):
        return [output.loss() for output in self.ModelOutputs]
    
    def get_metrics(self):
        metriclist = [output.get_metrics() for output in self.ModelOutputs]
        
        # at the moment, they need to be all lists, or all dicts
        if type(metriclist[0]) is list:
            metrics = []
            for mm in metriclist:
                metrics.extend(mm)
        elif isinstance(metriclist[0],dict):
            metrics = {}
            for mm in metriclist:
                metrics = {**metrics, **mm}
        
        return metrics
    
    def compile(self,learning_rate=1e-3,**kwargs):
        opt = self.OptimizerFunction(learning_rate)
        
        losses = self.get_losses()
        
        metrics = self.get_metrics()
        
        if self.LossWeights is not None:
            kwargs = {'loss_weights':self.LossWeights, **kwargs}
            
        self.Model.compile(opt,losses, metrics=metrics,**kwargs)
        
        return self
    
    def add_fixed_scheduler(self,epochs=[20,40,80],factors=[0.2,0.4,0.5]):
        fdict = defaultdict(lambda:1.0,{x:y for x,y in zip(epochs,factors)})
        def scheduler(lr,epoch):
            # need to check the order of the inputs above
            return lr*fdict[epoch]
        
        self.Callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule))
        
        return self
        
    def add_checkpoint(self,format_string='model_checkpoint{epoch}.h5',period=10,**kwargs):
        
        filepath = os.path.join(self.OutputFolder,format_string)
        cbk = tf.keras.callbacks.ModelCheckpoint(
            filepath, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=False,
            save_weights_only=False, 
            mode='auto', 
            save_freq='epoch',
            period=period,
            options=None, **kwargs
        )
        
        self.Callbacks.append(cbk)
        
        return self
    
    def add_logger(self,name='Log.csv',append=True,**kwargs):
        filepath = os.path.join(self.OutputFolder,name)
        
        cbk = tf.keras.callbacks.CSVLogger(
            filepath,append=append,**kwargs
        )
        
        self.Callbacks.append(cbk)
        
        return self
    
    
    def run_generator(self,gen,vgen=None,epochs=10,epoch_length=200, learning_rate=None,**kwargs):
        
        if learning_rate is not None:
            self.compile(learning_rate)
        
        fit_kw = {'epochs':epochs, 'steps_per_epoch':epoch_length, 'verbose':1, **kwargs}
        if vgen is not None:
            fit_kw = {**fit_kw, 'validation_data':vgen, 'validation_steps':epoch_length//20}
        
        h = self.Model.fit(
            gen, **fit_kw
        )
        
        return h
    
    def run_numpy(self,x,y,vx=None,vy=None,epochs=10, learning_rate=None,**kwargs):
        
        """
        Here might want to shuffle or augment between epochs?  Although why not just
        do a generator in that case
        """
        
        if learning_rate is not None:
            self.compile(learning_rate)
        
        fit_kw = {'epochs':epochs, 'verbose':1, **kwargs}
        if vx is not None:
            fit_kw = {**fit_kw, 'validation_data':(vx,vy)}
        
        h = self.Model.fit(
            gen, **fit_kw
        )
        
        return h
    
    
        
        
    
    
    
        