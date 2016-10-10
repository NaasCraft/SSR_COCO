#coding=utf-8

'''
    authors : R. Canyasse, C. Sutton, G. Demonet
    last update : 30/08/16 by RC
    
    description :
        Preprocessing functions used in the Keypoints COCO Challenge
        
    list of content :
        + classes :
            - Preprocessor
            
        + functions :
            - resize_bbox
            - parameters_from_partition(keypoints_batch, partition)
            
'''

###########
# Imports #
###########

import numpy as np
from copy import copy, deepcopy
#import copy
from skimage.transform import rescale

####################
# Global variables #
####################



#############
# Functions #
#############

def resize_bbox(sub_i,output_shape):
    """
    input : sub_image / output shape (3 dimension x,y,channels as it is in coco)
    output : the resized image to the desired shape, we respect the ratio so the image is centered
    """
    assert len(output_shape) == 3, "not shape for coco image"
    assert output_shape[2] == 3, "need 3 channels"
    
    image = copy(sub_i)
    #ann = copy.deepcopy(sub_ann)
    
    if type(image)== np.ndarray and image.shape[2]==3: #we don't deal with images that are not with three channels 3
        input_shape = image.shape

        if input_shape[0] > input_shape[1] : #height greater than width
            scale = float(output_shape[0]) / input_shape[0]  #important to take float division and not integer division !
        else:
            #width greater than height
            scale = float(output_shape[1]) / input_shape[1]

        res_I = rescale(sub_i,scale)
        image_out = np.zeros(output_shape)

        x_decay = np.floor(output_shape[1]/2 - res_I.shape[1]/2)
        y_decay = np.floor(output_shape[0]/2 - res_I.shape[0]/2)


        image_out[y_decay:res_I.shape[0]+y_decay ,x_decay:res_I.shape[1]+x_decay , :] = res_I

        return image_out
    else:
        pass

def make_it_random(function, prob=0.5):
    '''
    returns original function with probability `prob`,
    else returns identity
    '''
    if np.random.rand() <= prob:
        return function
    else:
        return lambda x: x


def format_kp_labels(keypoints_batch, partition, new_archi=False):
    """ 
    input : 
        keypoints_batch: batch of vector of keypoints 
            of (x_i,y_i,v_i) i=0...17, ndarray of list
        
        partitions : list of list describing the 
            rearrangement of keypoints for models

    
    outputs : 
        - dict where keys are output_layers_name 
                 and values the batch for relevant columns
    """
    
    detection_name = "output_detection_"
    regression_name = "output_regression_"
    
    output_dict = {}
    
    if not new_archi: #initial specialists v1->v3 behavior
        
        for p in range(len(partition)): 
            # get the keypoints in this sub partition as np array
            kp = np.asarray(partition[p]) 

            # we sort the keypoints inside a part of a partition
            kp = np.sort(kp) 

            # extract indexes for x, y, and v values
            x_idx = 3*kp 
            y_idx = 3*kp + 1
            v_idx = 3*kp + 2

            # sort final indexes
            kp_idx = np.sort(np.append(x_idx, y_idx))
            v_idx.sort()

            # batch for v in this partition
            output_dict[detection_name+str(p)] = keypoints_batch[:, v_idx]

            # batch for kp (x,y) in this partition
            output_dict[regression_name+str(p)] = keypoints_batch[:, kp_idx]
    
    else: #new USSR model : Unique Stacked Specialist Regressors
        
        for name, kp in new_archi.iteritems():
            
            xi, yi = [3*kp + i for i in range(2)]
            kpi = np.sort(np.append(xi, yi))
            output_dict[name] = keypoints_batch[:, kpi]
    
    return output_dict

def extract_subim_kp(I, bboxList, kpList):
    '''
    extract a subimage and adapt keypoints
    could be used in bbox_extraction below (todo)
    '''
    try:
        y, x, w, h = [int(el) for el in bboxList]
        
        if len(I.shape)==2:
            img = I.reshape((I.shape[0],I.shape[1],1))
        else:
            img= I
            
        img = img[x:x+h,y:y+w,:]
        
        kp_ix = [(3*ix, 3*ix+1, 3*ix+2) for ix in range(17)]
        kp = copy(kpList)
        
        for yi, xi, vi in kp_ix:
            if kp[vi] > 0:                   
                kp[yi] -= y
                kp[xi] -= x
                
        return img, kp
        
    except:
        print(I.shape)
        print('Smthg went wrong in subimage extraction for kp.')
        raise

def bbox_extraction(I, anns):
    """
    This function aims to take bbox separatedly
    input: an image (matrix) and its annotation (dict)
    output: list of sub-images (ndarray) and their adapted notation for the showAnns function  
    """
    # it is important to work on copies (deep copies is for dictionnary to copy all and not only the first layer)
    I = copy(I)
    anns = deepcopy(anns)

    image_list = [] 
    ann_list = []
    
    if len(anns)>0:
        for i in range(len(anns)):
            
            #dealing with images

            y = anns[i]["bbox"][0] #seems to have an error in the documentation of coco
            x = anns[i]["bbox"][1]
            w = anns[i]["bbox"][2]
            h = anns[i]["bbox"][3]
            image_list.append(I[x:x+h,y:y+w,:])
            
            #dealing with annotations
            ann_tmp = []
            ann_tmp.append(deepcopy(anns[i]))
            ann_tmp[0][u'segmentation'] = [[0,0]]  #remove segmentation that is hard to adapt, and useless for the eval
            for j in range(0,51,3) :
                if ann_tmp[0][u'keypoints'][j+2]>0:                    
                    ann_tmp[0][u'keypoints'][j] = ann_tmp[0][u'keypoints'][j] - y
                    ann_tmp[0][u'keypoints'][j+1] = ann_tmp[0][u'keypoints'][j+1] - x 
            ann_list.append(ann_tmp)
    return image_list, ann_list

def horizontal_flip(I,anns):
    """ 
    input : inputs are the elements of of the previous function bbox_extraction i.e. an sub image and the corresponding adapted information
    output : image horizontaly flipped and anntation adapted 
    """
    #dealing with images
    I = copy.copy(I)    
    I = I[:,::-1,:]
    
    #dealing with annotations
    ann = copy.deepcopy(anns)

    w =  ann[0][u'bbox'][2]
    for j in range(0,51,3) :
        if ann[0][u'keypoints'][j+2]>0:                    
            ann[0][u'keypoints'][j] = w - ann[0][u'keypoints'][j]
        
    return I, ann

def rescale_kp2(image,kpList,scale):
    """
    For the image do the same rescale as the one in scikit image
    For the annotation, adapt the keypoints
    
    input : sub_image, sub_annotation and de desired scale (float is ok)
    output : scaled image,annotations
    """
    image = copy(image)    
    image_out = rescale(image,scale)
    
    kp_ix = [(3*ix, 3*ix+1, 3*ix+2) for ix in range(17)]
    kp = copy(kpList)

    for yi, xi, vi in kp_ix:
        if kp[vi] > 0:                   
            kp[yi] *= scale
            kp[xi] *= scale
        
    return image_out, kp

def resize_kp2(img,kpList,output_shape):
    """
    input : sub_image, s for bbox extraction / output shape (3 dimension x,y,channels as it is in coco)
    output : the resized image to the desired shape, we respect the ratio so the image is centered
    """
    assert len(output_shape) == 3, "not shape for coco image"
    assert output_shape[2] == 3, "need 3 channels"
    
    image = copy(img)
    kp = copy(kpList)
    
    if type(image)== np.ndarray and image.shape[2]==3: #we don't deal with images that are not with three channels 3
        input_shape = image.shape

        if input_shape[0] > input_shape[1] : #height greater than width
            scale = float(output_shape[0]) / input_shape[0]  #important to take float division and not integer division !
        else:
            #width greater than height
            scale = float(output_shape[1]) / input_shape[1]


        res_I, res_kp = rescale_kp2(image,kp,scale) #up or downsampling

        image_out = np.zeros(output_shape)

        x_decay = int(np.floor(output_shape[1]/2 - res_I.shape[1]/2))
        y_decay = int(np.floor(output_shape[0]/2 - res_I.shape[0]/2))


        image_out[y_decay:int(res_I.shape[0])+y_decay,
                  x_decay:int(res_I.shape[1])+x_decay, :] = res_I
        
        kp_ix = [(3*ix, 3*ix+1, 3*ix+2) for ix in range(17)]
        
        for yi, xi, vi in kp_ix:
            if res_kp[vi] > 0:                   
                res_kp[yi] += x_decay
                res_kp[xi] += y_decay

        return image_out, res_kp
    else:
        raise Exception("we don't deal with images that are not with three channels 3")
        
def rescale_kp(image,annotation,scale):
    """
    For the image do the same rescale as the one in scikit image
    For the annotation, adapt the keypoints
    
    input : sub_image, sub_annotation and de desired scale (float is ok)
    output : scaled image,annotations
    """
    image = copy(image)
    ann = deepcopy(annotation)
    
    print(ann)
    #print(type(image))
    #print(image)
    
    image_out = rescale(image,scale)
    
    for j in range(0,51,3):
        if ann[0][u'keypoints'][j+2]>0:                    
            ann[0][u'keypoints'][j]   = scale * ann[0][u'keypoints'][j]
            ann[0][u'keypoints'][j+1] = scale * ann[0][u'keypoints'][j+1]
        
    return image_out, ann

def resize_kp(sub_i,sub_ann,output_shape):
    """
    input : sub_image, sub_annotation for bbox extraction / output shape (3 dimension x,y,channels as it is in coco)
    output : the resized image to the desired shape, we respect the ratio so the image is centered
    """
    assert len(output_shape) == 3, "not shape for coco image"
    assert output_shape[2] == 3, "need 3 channels"
    
    image = copy(sub_i)
    ann = deepcopy(sub_ann)
    
    if type(image)== np.ndarray and image.shape[2]==3: #we don't deal with images that are not with three channels 3
        input_shape = image.shape

        if input_shape[0] > input_shape[1] : #height greater than width
            scale = float(output_shape[0]) / input_shape[0]  #important to take float division and not integer division !
        else:
            #width greater than height
            scale = float(output_shape[1]) / input_shape[1]


        res_I, res_ann = rescale_kp(image,ann,scale) #up or downsampling

        image_out = np.zeros(output_shape)

        x_decay = np.floor(output_shape[1]/2 - res_I.shape[1]/2)
        y_decay = np.floor(output_shape[0]/2 - res_I.shape[0]/2)


        image_out[y_decay:res_I.shape[0]+y_decay,
                  x_decay:res_I.shape[1]+x_decay, :] = res_I

        for k in range(0,51,3):
            if res_ann[0][u'keypoints'][k+2]>0:                    
                res_ann[0][u'keypoints'][k]   = res_ann[0][u'keypoints'][k] + x_decay
                res_ann[0][u'keypoints'][k+1] = res_ann[0][u'keypoints'][k+1] + y_decay

        return image_out, res_ann
    else:
        raise Exception("we don't deal with images that are not with three channels 3")

def get_list_of_bbox(Ids, presence_vector):
    """
    If you want to have the list of bbox containing at least the keypoint of the keypoint vector
    input : ImgIds, id of images in the database
            presence_vector ; keypoints that are in the bbox 
    output:
    List of bboxes that contain at least keypoints of the presence vector
    """
    kp_idx = np.arange(2,51,3)[presence_vector==1] # indices to check
    result = []
    for i in range(len(Ids)) :
        img = coco.loadImgs(Ids[i])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns)>0: 
            bbox_idx = []
            for j in range(len(anns)) : 
                kp_array = np.asarray(anns[j]["keypoints"])
                if np.sum(kp_array[kp_idx]==2) == np.sum(presence_vector):
                    bbox_idx.append(j)

            if len(bbox_idx)>0:
                result.append((Ids[i],bbox_idx))
    return result

def get_X_y(image_path, output_shape, presence_vector, qty = 0.1):
    """
    This function builds the training/val set for the neural net
    
    input :  the list of selected bbox of the previous function  
    output : X and y data for the fit function of keras sequential
    """
     
    imgIds = coco.getImgIds(catIds=catIds)
    split_index = int(np.floor(qty*len(imgIds)))
    list_of_bbox = get_list_of_bbox(imgIds[:split_index], presence_vector)
    
    X = []
    y_v = []
    y_kp = []
    anns = []
    
    
    #vector of indices to select or x, or y or (x,y) or v in the keypoint vector
    x_idx = 3*np.arange(17)[presence_vector==1]
    y_idx = 3*np.arange(17)[presence_vector==1] + 1
    kp_idx = np.sort(np.append(x_idx, y_idx)) 
    v_idx = 3*np.arange(17)[presence_vector==1] + 2
    
    for res in list_of_bbox :
        # good format of the id
        ind = res[0]
        id_good = (12-len(str(ind)))*'0' + str(ind)
        #get image / ann
        I = plt.imread(image_path+id_good+".jpg")

        if check_image(I) : 
            
            annId = coco.getAnnIds(imgIds=ind, catIds=[1], iscrowd=None)
            ann = coco.loadAnns(annId)

            # get sub images and sub annotations

            sub_I_list, sub_ann_list = bbox_extraction(I, ann)

            
            # rescale to the output shape
            for k in res[1]:

                sub_I_resized, sub_ann_resized = resize_kp(sub_I_list[k],
                                                           sub_ann_list[k],
                                                           output_shape)
                X.append(sub_I_resized)
                y_kp.append(np.asarray(\
                     sub_ann_resized[0][u'keypoints'])[kp_idx])  #(x,y)
                y_v.append(np.asarray(\
                     sub_ann_resized[0][u'keypoints'])[v_idx])   #v 
                #print( sub_ann_resized )
                #plt.figure()
                #plt.imshow(sub_I_resized)
                #coco.showAnns(sub_ann_resized)
                anns.append(sub_ann_resized)
                
    X = np.asarray(X)
    y_kp = np.asarray(y_kp)
    y_v = np.asarray(y_v)
    anns = np.asarray(anns)
    return X, y_kp, y_v, anns

def check_image(I) :
    """
    check if the inputs have good features for the nn
    """    
    if type(I)==np.ndarray :
        #print("type ok")
        pass
    else :
        return False
    
    if len(I.shape) == 3 : 
        #print("shape length 3 ok")
        pass
    else:
        return False
    
    if I.shape[2]==3:
        #print("3 channels ok")
        pass
    else:
        return False
    
    return True

def y_to_kp(y_kp,y_v,presence_vector): 
    res = np.zeros(51)
    
    x_idx = 3*np.arange(17)[presence_vector==1]
    y_idx = 3*np.arange(17)[presence_vector==1] + 1
    kp_idx = np.sort(np.append(x_idx, y_idx)) 
    v_idx = 3*np.arange(17)[presence_vector==1] + 2

    res[kp_idx] = y_kp
    res[v_idx] = y_v
    
    return res


def rescaleKP_toBBox(keypoints, bbox, shape=(256,256)):
    '''
        returns adapted kp to initial image, given bbox as [x, y, width, height]
    '''
    kp_ixs = [(3*k, 3*k+1) for k in range(17)]
    kpList = copy(keypoints)
    
    # bbox[2] is width, bbox[3] is height
    # shape[1] is width, shape[0] is height
    bW, bH = float(bbox[2]), float(bbox[3])
    sH, sW = shape
    
    scale_x = sW / bW
    scale_y = sH / bH 
    
    #scale = max(scale_x, scale_y) # il faut prendre le min
    scale = min(scale_x, scale_y)
    
    x_decay = (sW - scale*bW)/2.
    y_decay = (sH - scale*bH)/2.
    
    if x_decay < 1.: x_decay = 0.
    if y_decay < 1.: y_decay = 0.
    
    for (x_ix,y_ix) in kp_ixs:
        if ( kpList[x_ix] > 0 or kpList[y_ix] > 0  ):

            kpList[x_ix] -= x_decay
            kpList[x_ix] /= scale
            kpList[x_ix] += bbox[0]
            kpList[x_ix] = round(kpList[x_ix], 2)

            kpList[y_ix] -= y_decay
            kpList[y_ix] /= scale
            kpList[y_ix] += bbox[1]
            kpList[y_ix] = round(kpList[y_ix], 2)
    
    return kpList

def rescaleKP_toBBox_regOnly(keypoints, bbox, shape=(256,256)):
    '''
        returns adapted kp to initial image, given bbox as [x, y, width, height]
    '''
    kp_ixs = [(2*k, 2*k+1) for k in range(17)]
    kpList = copy(keypoints)
    
    # bbox[2] is width, bbox[3] is height
    # shape[1] is width, shape[0] is height
    bW, bH = float(bbox[2]), float(bbox[3])
    sH, sW = shape
    
    scale_x = sW / bW
    scale_y = sH / bH 
    
    #scale = max(scale_x, scale_y) # il faut prendre le min
    scale = min(scale_x, scale_y)
    
    x_decay = (sW - scale*bW)/2.
    y_decay = (sH - scale*bH)/2.
    
    for (x_ix,y_ix) in kp_ixs:
        
        kpList[x_ix] -= x_decay
        kpList[x_ix] /= scale
        kpList[x_ix] += bbox[0]
        kpList[x_ix] = round(kpList[x_ix], 2)
        
        kpList[y_ix] -= y_decay
        kpList[y_ix] /= scale
        kpList[y_ix] += bbox[1]
        kpList[y_ix] = round(kpList[y_ix], 2)
    
    return kpList



###########
# Classes #
###########

class Preprocessor():
    '''
        Define a preprocessor class with standard functions,
        that can apply them altogether with .run()
        and can return a history of changes with .history()
    '''
    
    def __init__(self, params):
        
        self.corresp = {'hflip': horizontal_flip,
                        'resize': resize_kp2}
        
        if params.has_key('ignore') and params['ignore']:
            self.ignore = True
        else:
            self.ignore = False
        
        if not self.ignore:
            try:
                self.steps = params['steps']
                self.resize = params['resize']
                
            except KeyError:
                print("Preprocessing parameters are in wrong format.")
                raise
            
    
    def run(self, x, y):
        
        x_prep, y_prep = copy(x), copy(y)
        
        if not self.ignore:
            # do the preprocessing
            for step in self.steps:

                try:
                    func = self.corresp[step['name']]
                    
                    if step['name'] == 'resize':
                        func2 = copy(func)
                        func = lambda x,y: func2(x,y,self.resize)

                except KeyError:
                    print("Unknown step for preprocessing")
                    raise

                else:
                    if step.has_key('random') and step['random']:
                        func = make_it_random(func)
                    
                    #print('x_prep.shape : {}'.format(x_prep.shape))
                    #print('y_prep : {}'.format(y_prep))
                    x_prep, y_prep = func(x_prep, y_prep)
        
        return x_prep, y_prep
