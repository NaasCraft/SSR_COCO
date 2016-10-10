###########
# Imports #
###########

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras.layers import Input, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten ,Convolution2D
from keras.models import Model


def speModel_VGG_(partition,verbose = True, **kwargs):
    
    # base model parameters
    
    regression_fm_name = 'vgg16'
    
    if kwargs.has_key('base_model'):
        if kwargs['base_model'] in [ 'vgg19'] : 
            regression_fm_name = 'vgg19'

    if verbose :
        print 'The model is based on : '+ regression_fm_name
    
    # image shape
    
    if kwargs.has_key('image_shape'):
        image_shape = kwargs['image_shape']
    else:
        image_shape = (256,256,3)
        
    
    main_input = Input(shape=image_shape, name='main_input')
    
    if regression_fm_name == 'vgg19':
        vgg = VGG19(weights='imagenet',include_top=False,input_tensor=main_input)
    else:
        vgg = VGG16(weights='imagenet',include_top=False,input_tensor=main_input)
    if verbose:
        print ' Base model loaded '
            
            
    # trainable ?
    
    if kwargs.has_key('trainable'):
        trainable = kwargs['trainable']
        vgg.trainable = bool(trainable)
    else:
        vgg.trainable = False
        
    #if verbose:
        #print 'vgg will be trained'
    vgg_out = vgg(main_input)
    
    outputs = []
    
    # loss parameters
    loss_dict = {}
    loss_weights = {}
    
    for p in range(len(partition)):
        
        base = 'block{}_'.format(p)
        
        conv1 = Convolution2D(256, 3, 3,border_mode = 'same',activation = 'relu',init = 'glorot_normal',name = base + 'conv1')(vgg_out)
        
        pool1 = MaxPooling2D((2,2), strides = (2,2),
                             name = base + 'pool1')(conv1)
        
        conv2 = Convolution2D(256, 3, 3, 
                              border_mode = 'same', 
                              activation = 'relu',
                              name = base + 'conv2',
                              init = 'glorot_normal')(pool1)
        
        pool2 = MaxPooling2D((2,2), strides=(2,2),
                             name = base + 'pool2')(conv2)
        
        flat = Flatten(name = base + 'flat')(pool2)
        
        dense1 = Dense(2048, 
                       activation = 'relu',
                       name = base + 'dense1',init = 'glorot_normal')(flat)
        
        drop1 = Dropout(.5, name = base + 'drop1')(dense1)
        
        dense2 = Dense(2048, 
                       activation = 'relu',
                       name = base + 'dense2',init = 'glorot_normal')(drop1)
        
        drop2 = Dropout(.5, name = base + 'drop2')(dense2)
        
        out_reg = Dense(2 * len(partition[p]), 
                        activation = 'relu', 
                        name = 'output_regression_'+str(p),init = 'glorot_normal')(drop2)

        # loss parameters
        #loss_dict['output_regression_'+str(p)] = 'sum_squared_error_coco' 
        #loss_weights['output_regression_'+str(p)] = 1.

        # adding the output layer
        outputs.append(out_reg) # same warning as detection
        
    
    #instanciate the 1 input - multi-outputs model
    return Model(input=main_input, output=outputs)    
    
    #model.compile(optimizer= 'adagrad', loss = loss_dict, loss_weights = loss_weights)    
                


    
    