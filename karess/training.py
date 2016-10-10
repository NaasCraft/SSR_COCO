import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping


class OutputSaver(keras.callbacks.Callback):
    '''
        Keras callback used to store model outputs during training, 
        on a specific set of validation inputs. These have to be
        specified at callback initialization.
    '''
    
    def __init__(self, img_batch, model_name='', verbose=False):
        super(OutputSaver, self).__init__()
        
        # store img batch in memory
        ##TODO : check shape and other kind of potential errors
        self.img_batch = img_batch
        
        self.verb = bool(verbose)
        
        if model_name != '':
            self.model_name = str(model_name)
        else:
            self.model_name = 'tmp_model_outputsave'
    
    def on_train_begin(self, logs={}):
        import os, warnings
        os.chdir('/home/ubuntu/coco/models/output_save/')
        
        model_name = self.model_name
        
        if os.path.exists(model_name):
            warnings.warn('{} folder already exists. Resetting it...'.format(model_name))
            os.system('sudo rm -rf {} && mkdir {}'.format(model_name, model_name))
        else:
            os.system('mkdir {}'.format(model_name))
    
    def on_epoch_begin(self, epoch, logs={}):
        pass
    
    def on_epoch_end(self, epoch, logs={}):
        import os, pickle
        os.chdir('/home/ubuntu/coco/models/output_save/{}/'.format(self.model_name))
        
        if self.verb: print('\n---\nPredicting for callback batch...')
        output = self.model.predict_on_batch(self.img_batch)
        
        if self.verb: print('Done.\n\nSaving output at "output_{}"...'.format(epoch))
        
        with open('output_{}'.format(epoch), 'wb') as dumpfile:
            pickle.dump(output, dumpfile)
            
        if self.verb: print('Done.\n---\n')
    
    def on_train_end(self, logs={}):
        pass

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.time = 0
        
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.time += 1
        self.losses.append(loss)
        '''
        plt.plot(range(self.time),self.losses)
        display.clear_output(wait=True)
        display.display(plt.gcf())'''