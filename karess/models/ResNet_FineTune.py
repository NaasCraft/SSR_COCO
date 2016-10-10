###########
# Imports #
###########

from keras.applications.resnet50 import ResNet50

from keras.layers import Input, Dropout
from keras.layers import Dense, Flatten 
from keras.models import Model



def ResNet_FineTune_(partition,trainable,verbose =True, **kwargs):
    
    
    if kwargs.has_key('image_shape'):
        image_shape = kwargs['image_shape']
    else:
        image_shape = (256,256,3)
        
    
    main_input = Input(shape=image_shape, name='main_input')

    resnet = ResNet50(weights='imagenet',
                    include_top=False,
                    input_tensor=main_input)
    
    resnet.trainable = bool(trainable)    
    if trainable :
        print 'The model is based on ResnNet50. It will be fine tuned'
    else :
        print 'The model is based on ResnNet50. It will not be fine tuned'

    resnet_out = resnet(main_input)
    
    outputs = []
    
    #########################################
    for p in range(len(partition)):
        
        base = 'block{}_'.format(p)

        flat = Flatten(name = base + 'flat')(resnet_out)
        
        dense1 = Dense(2048,activation = 'relu',init = 'glorot_normal',name = base + 'dense1')(flat)

        drop1 = Dropout(.5, name = base + 'drop2')(dense1)

        out_reg = Dense(2 * len(partition[p]),activation = 'relu',init = 'glorot_normal',name = 'output_regression_'+str(p))(drop1)
        outputs.append(out_reg)
    return Model(input=main_input, output=outputs)
        