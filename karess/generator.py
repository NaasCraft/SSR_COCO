#coding=utf-8

'''
    authors : R. Canyasse, C. Sutton, G. Demonet
    last update : 30/08/16 by RC
    
    description :
        Python generator for loading image batches in Keras.
        
    list of content :
        + classes :
            - BatchCOCO
            
        + functions :
            - resize_bbox
            - load_anns
            - strip_objs
            - get_anns_by_id
            - load_img
            - load_sample
            - gen_batch_from_file
            - concatKP
            - showBatch
            
            - load_sample_from_bbox
            - gen_batch_from_bbox_file
'''

###########
# Imports #
###########

import os, sys
import os.path as osp
import karess.preprocessing as kprep
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from matplotlib.pyplot import imread
from pycocotools.coco import COCO
from warnings import warn
import re
import skimage.io as io

####################
# Global variables #
####################

# directory paths
CURRDIR = osp.abspath('.')
# to put in module : 
# CURRDIR = osp.dirname(__file__)
DATADIR = osp.abspath('/home/ubuntu/coco/data')
TRAINDATA = osp.join(DATADIR, 'train2014')
TRAINVALDATA  = osp.join(DATADIR, 'train2014')
VALDATA = osp.join(DATADIR, 'val2014')
TESTDATA = osp.join(DATADIR, 'test2015')
ANNDIR = osp.join(DATADIR, 'annotations')
dirs = [TRAINDATA,VALDATA,TESTDATA,TRAINDATA,TRAINDATA]

# file paths
filenameTP = "person_keypoints_{}.json"
modes = ['train', 'val', 'test','trainval','trainvalbigbbox']
mode_date = ['train2014', 'val2014', 'test2015','trainval2014','trainval2014']

filenames = [osp.join(ANNDIR,
                      filenameTP.format(el))\
             for el in mode_date]
annFiles = dict([el for el in zip(modes,filenames)])
dataDirs = dict([el for el in zip(modes,dirs)])

# init empty annotation files
anns = {}.fromkeys(modes)

#############
# Functions #
#############



def load_anns(mode, unload=False):
    '''
        Loads (or unloads) mode annotations in memory.
        mode is either 'train', 'val', or 'test'.
    '''
    if not unload:
        if not anns[mode]:
            anns[mode] = COCO(annFiles[mode])
        else:
            msg = '{} annotations are already loaded.'
            warn(msg.format(mode))
    else:
        anns[mode] = None

def strip_objs(annotations, objs):
    '''
        Helper to process annotations format and make it readable
        for Keras. Will surely be improved.
    '''
    if objs=='all':
        return annotations
    elif objs=='bbox' or objs=='keypoints':
        return [el[objs] for el in annotations]

def get_anns_by_id(annFile, img_id, objs='all'):
    '''
        Returns list of annotations for given img id.
        objs can be :
            + 'all' for the full dict
            + 'bbox'
            + 'keypoints'
    '''
    try:
        catIds = annFile.getCatIds()
        annIds = annFile.getAnnIds(imgIds=img_id,
                                   catIds=catIds)
        
        annotations = annFile.loadAnns(annIds)
        return strip_objs(annotations, objs)
        
    except Exception as err:
        msg='Couldn\'t load annotations for img #{}'
        print(msg.format(img_id))
        print('Error msg was :\n\t'.format(err))
        
def load_img(path, grayscale=False, target_size=None):
    '''
        image of shape (height, width, channels)
    '''
    print("nooooooo")
    # following from keras/preprocessing/image.py
    from PIL import Image
    
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    elif len(x.shape)!= 3:
        raise Exception('Unsupported image shape: ', x.shape)
    
    return x

def load_sample(line, preprocessor,
                mode='train', objs='all', with_kp=False):
    '''
        Takes a line from .txt description, which should contain
        at least the img id and its filename
        __More__ :
        This .txt file could also contain info about the number of
        persons in the image and maybe the annotation ids.
    '''
    #print 'mode is : ' +mode


    args = line.split(' ')
    
    if with_kp :
        ann_id, img_id, img_name = args[:3]
        ann_id = int(ann_id)
    if mode == 'test' :
        
        return load_sample_from_bbox(line)

    img_id = int(img_id)
    
    if not anns[mode]:
        load_anns(mode)
    
    annFile = anns[mode]
    
    # specify proper path for img
   
    img_path = osp.join(dataDirs[mode], img_name)
    
    x = imread(img_path)
    
    # brutally ignore grayscale atm
    if len(x.shape)==2:
        return None, None
    
    else:
        if with_kp:
            y_ann = annFile.loadAnns(ann_id)[0]
            x, y = kprep.extract_subim_kp(x, y_ann['bbox'], 
                                          y_ann['keypoints'])
        else:
            y = get_anns_by_id(annFile, img_id, objs)

        try:
            x, y = preprocessor.run(x, y)
        except:
            return None, None
        else:
            if mode == 'test': y = None
                
            return x, y
        
def to_rgb(im):
    print 'from grayscale to rgb'
    # I would expect this to be identical to 1a
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
    return ret
        
def load_sample_from_bbox(line):
    
    
    '''
        Takes a line from .txt description, which contains
        at the img id and its filename and the bbox coordinates
        __More__ :

    '''
    
    #print line
    args = line.split("\t")
    #print args
    img_id, img_name = args[:2]

    #I = load_img('/home/ubuntu/coco/data/test2015/%s'%(img_name), grayscale=False, target_size=None)
    #I = io.imread('/home/ubuntu/coco/data/test2015/%s'%(img_name))
    I = imread('/home/ubuntu/coco/data/test2015/%s'%(img_name))
    if (len(I.shape) == 2) :
        I = to_rgb(I)
    else:
        if (len(I.shape) == 3 and I.shape[2] == 1) :
            I = to_rgb(I)
    
    x,y,h,w = np.float64(args[2:6])
    
    resized_bbox = kprep.resize_bbox(I[y:y+w,x:x+h,:],(256,256,3))
      
    #return I[y:y+w,x:x+h,:]
    return resized_bbox


def gen_batch_from_file(path, samples_per_epoch, **kwargs):
    '''
        kwargs definition todo
        --> allow for multiple (in/out)puts with dicts
            and names
    '''
    
    ############### 
    # extract parameters from kwargs
    #     --> if time, do this with a loop/function
    
    # preprocessing parameters
    if kwargs.has_key('preprocess_params'):
        prep_params = kwargs['preprocess_params']
    else:
        prep_params = {'ignore':True}
        
    # batch parameters
    if kwargs.has_key('batch_params'):
        batch_params = kwargs['batch_params']
    else:
        batch_params = {}
        
    # verbosity (for now, True or False)
    if kwargs.has_key('verbose'):
        verbose = kwargs['verbose']
    else:
        verbose = False
    
    if kwargs.has_key('objs'):
        objs = kwargs['objs']
    else:
        objs = 'all'
        
    if kwargs.has_key('mode'):
        mode = kwargs['mode']
    
    else:
        mode = 'train'
    print 'mode is ' + mode    
    if 'new_archi' in kwargs:
        new_archi = kwargs['new_archi']
    else:
        new_archi = False
    
    if kwargs.has_key('shuffle'):
        shuffle = bool(kwargs['shuffle'])
    else:
        shuffle = True
    #
    ###############
    
    while 1:
        # get list
        
        if verbose: print("Shuffling data...")
        with open(path, 'r') as source:
            data = [(random.random(), line) for line in source]
        
        # randomize it
        if shuffle: data.sort()
        
        total_ix = 0 #count total samples loaded in epoch
        batch_params['mode'] = mode #specify mode in batch_params
        batch = BatchCOCO(batch_params,verbose) #instance of BatchCOCO object
        with_kp = 1 if batch.kp else 0
        
        if verbose: print("Loading batch...")
        for _, line in data:
            # initialize preprocessor
            if with_kp: # force params if kp mode
                prep_params['ignore'] = False
                prep_params['steps'] = [{'name':'resize'}]
                prep_params['resize'] = (batch.resize[0],
                                         batch.resize[1], 3)
                    
            prep = kprep.Preprocessor(prep_params)
            
            # get sample from line
            ## Note :
            if mode=='test' :
                new_x =  load_sample_from_bbox(line)
            else :    
                new_x, new_y = load_sample(line, prep,  
                                       mode, objs, with_kp)
            
            if new_x is None:
                break
            
            # update current nb of samples stored
            # with .update()
            ## Note: if mode=='test', .update() ignores new_y
            ## RC : The previous note is false, I needed to change a few things,please test your code..
            
            if mode == 'test' :
                new_y = 'None'
            batch.update(new_x, new_y)
            total_ix += 1
            
            if batch.fill_state==batch.size:
                # .sd_reset() has to
                # 1) generate proper keras format for a batch
                # 2) clear the batch object
                if verbose: print("Batch loaded.")
                yield batch.yield_reset()
                sample_ix = 0
                
                #now check that there are enough imgs left
                #to make a batch
                remaining = len(data) - total_ix
                if remaining < batch.size:
                    warn('Not enough remaining samples '
                         'to load a batch. Loading file again.')
                    print("There are {} samples in {}".format(
                        len(data), path))
                    break 
            
            if total_ix > samples_per_epoch:
                warn('Epoch comprised more than '
                     '`samples_per_epoch` samples, '
                     'which might affect learning results. '
                     'Set `samples_per_epoch` correctly '
                     'to avoid this warning.')
            if total_ix >= samples_per_epoch:
                #this should break the for loop and load file again
                break

import re

def concatKP(outputDict, partition):
    
    try:
        batch_size = outputDict['output_detection_0'].shape[0]
    except:
        print('Output dictionary in wrong format.')
        raise
    else:
        batchKPList = np.zeros((batch_size, 51))

        for key in outputDict:

            num = int(re.search('_([0-9]+?)$', key).group(1))
            style = key[7:10]
            
            # define indexes from num and partition and style
            kp_ixs = partition[num]
            
            if style=='det':
                kp_ixs = [3*k+2 for k in kp_ixs]
            else:
                kp_ixs = [(3*k, 3*k+1) for k in kp_ixs]
            
            for i in range(batch_size):
                kpList = outputDict[key][i]
                lenIx = len(kp_ixs)
                
                checkSize = len(kpList)==lenIx or len(kpList)==2*lenIx
                
                assert checkSize, 'Problem with kp list size'
                
                for j in range(len(kp_ixs)):
                    if style=='det':
                        batchKPList[i, kp_ixs[j]] = kpList[j]
                    else:
                        y_ix, x_ix = kp_ixs[j]
                        y_kp, x_kp = kpList[2*j:2*j+2]
                        batchKPList[i, y_ix] = y_kp
                        batchKPList[i, x_ix] = x_kp
                        #batchKPList[i, y_ix] = round(y_kp,2)
                        #batchKPList[i, x_ix] = round(x_kp,2)
                        
                
        return batchKPList
    
def concatKP_regOnly(outputDict, partition):
    
    try:
        batch_size = outputDict['output_regression_0'].shape[0]
    except:
        print('Output dictionary in wrong format.')
        raise
    else:
        batchKPList = np.zeros((batch_size, 51))

        for key in outputDict:

            num = int(re.search('_([0-9]+?)$', key).group(1))
            
            # define indexes from num and partition and style
            kp_ixs = partition[num]
            kp_ixs = [(3*k, 3*k+1) for k in kp_ixs]
            
            for i in range(batch_size):
                kpList = outputDict[key][i]
                lenIx = len(kp_ixs)
                
                checkSize = len(kpList)==2*lenIx
                
                assert checkSize, 'Problem with kp list size'
                
                for j in range(len(kp_ixs)):

                    y_ix, x_ix = kp_ixs[j]
                    y_kp, x_kp = kpList[2*j:2*j+2]
                    if x_kp >0 and y_kp >0 :
                        #batchKPList[i, y_ix] = y_kp
                        #batchKPList[i, x_ix] = x_kp
                        batchKPList[i, y_ix] = round(y_kp,2)
                        batchKPList[i, x_ix] = round(x_kp,2)
                        batchKPList[i, y_ix+1] = 1
                    else :
                        batchKPList[i, y_ix+1] = 0
                        
                
        return batchKPList

def showBatch(imgB, kpB=[], mode=False):
    batch_size = imgB.shape[0]
    if mode: # TODO
        load_anns('val')
        blankAnn = anns['val'].loadAnns(503453)[0]
    
    for i in range(batch_size):
        img = imgB[i]
        if mode: 
            kp_list = kpB[i]
            ann = {}.fromkeys(blankAnn)
            ann['keypoints'] = [int(x) for x in kp_list]
            ann['segmentation'] = [[0,0]]
            ann['category_id'] = 1
        
        plt.figure()
        plt.imshow(img)
        if mode:
            anns['val'].showAnns([ann])
        plt.show()
        
        yield ''

###########
# Classes #
###########

class BatchCOCO():
    '''
        Batch object to have some shared methods.
        __More__ :
        Use this class as a base for image batches and
        bbox subimage batches.
    '''
    def __init__(self, params, verbose=False):
        self.params = params
        
        if verbose: print('Initializing batch with following properties:')
        
        ############### 
        # extract parameters for batch
        #     --> if time, do this with a loop/function
        #
        if 'batch_size' in params:
            self.size = params['batch_size']
        else:
            self.size = 32
        if verbose: print('batch size: {}'.format(self.size))
            
        if 'image_size' in params\
                and len(params['image_size'])==2\
                and all(params['image_size']):
            self.resize = tuple(params['image_size'])
        else:
            self.resize = (256,256)
        if verbose: print('image input size: {}'.format(self.resize))
           
        if 'color_mode' in params:
            self.color = params['color_mode']
        else:
            self.color = 'rgb'
        self.n_channels = 3 if self.color=='rgb' else 1
        if verbose: print('color mode: {}'.format(self.color))
        
        if 'keypoints' in params:
            self.kp = params['keypoints']
            self.kp_partition = self.kp['partition']
            msg = 'keypoints partition: {}'.format(self.kp_partition)
            if verbose: print('--using keypoints \n'+msg)
        else:
            self.kp = None
            
        if 'mode' in params:
            self.mode = params['mode']
            msg = '---\n\n Current mode: {}'.format(self.mode)
            if verbose: print(msg)
        #
        ###############
        
        self.batch_x_size = (self.size, \
                             self.resize[0], self.resize[1],\
                             self.n_channels)
        
        self.batch_x = np.zeros(self.batch_x_size)
        
        if not self.mode == 'test':
            if self.kp: #in case of kp, we know batch_y shape
                self.batch_y = np.zeros((self.size,51))
            else:
                self.batch_y = []
            
        self.fill_state = 0
            
    def yield_reset(self, new_archi=False):
        # prep batch for keras
        
        if self.fill_state != self.size:
            msg = 'Tried to yield an incomplete batch.'
            raise Exception(msg)
        else:
            toYield = [{'main_input':copy(self.batch_x)}, {}]
            
            if self.mode != 'test':
                if self.kp:
                    toYield[1] = kprep.format_kp_labels(self.batch_y, 
                                                        self.kp_partition,
                                                        new_archi)
                else: #to improve for generalization
                    toYield[1]['main_output'] = copy(self.batch_y)
        # reset batch
        self.reset()
        # and return the properly formatted batch
        if self.mode == 'test': toYield = toYield[0]
        
        return toYield
    
    def update(self, x, y):
        # update batch storage
        if len(x.shape) != 3:
            msg = 'Cannot update batch with provided x, wrong shape'
            raise Exception(msg)
        else:
            self.batch_x[self.fill_state] = x
            
            if self.mode != 'test':
                self.batch_y[self.fill_state] = y
            
            self.fill_state += 1
        
        # and return current state
        # ie nb of samples stored
        return self.fill_state
    
    def update_test_batch(self, x):
        # update batch storage
        if len(x.shape) != 3:
            msg = 'Cannot update batch with provided x, wrong shape'
            raise Exception(msg)
        else:
            self.batch_x[self.fill_state] = x
            #self.batch_y[self.fill_state] = y
            self.fill_state += 1
        
        # and return current state
        # ie nb of samples stored
        return self.fill_state
    
    
    def reset(self):
        # reset storage but not params
        self.fill_state = 0
        self.batch_x = np.zeros(self.batch_x_size)
        
        if self.mode != 'test':
            if self.kp: #in case of kp, we know batch_y shape
                self.batch_y = np.zeros((self.size,51))
            else:
                self.batch_y = []
        
        pass