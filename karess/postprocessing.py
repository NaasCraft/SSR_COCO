#coding=utf-8

'''
    authors : R. Canyasse, C. Sutton, G. Demonet
    last update : 01/09/16 by RC
    
    description :
        PostProcessing functions used in the Keypoints COCO Challenge
        
    list of content :

            
        + functions :
            - resize_bbox
            - parameters_from_partition(keypoints_batch, partition)
            
'''

import re 
from copy import copy
import numpy as np


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

                    x_ix, y_ix = kp_ixs[j]
                    x_kp, y_kp = kpList[2*j:2*j+2]
                    if x_kp >0 and y_kp >0 :
                        #batchKPList[i, y_ix] = y_kp
                        #batchKPList[i, x_ix] = x_kp
                        batchKPList[i, y_ix] = round(y_kp,2)
                        batchKPList[i, x_ix] = round(x_kp,2)
                        batchKPList[i, y_ix+1] = 1
                    else :
                        batchKPList[i, y_ix+1] = 0
                        
                
        return batchKPList
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
    
    for (x_ix,y_ix) in kp_ixs:
        if ( kpList[x_ix] > 0 or kpList[y_ix] > 0  ):
            
            kpList[y_ix+1] = 1
            
            if kpList[x_ix] < x_decay: #bande noire gauche
                kpList[x_ix] = x_decay
            elif kpList[x_ix] > (sW-x_decay): #idem, droite
                kpList[x_ix] = sW - x_decay
            else:
                kpList[x_ix] -= x_decay
                
            kpList[x_ix] /= scale
            kpList[x_ix] += bbox[0]
            kpList[x_ix] = round(kpList[x_ix], 2)
            #kpList[x_ix] = int(round(kpList[x_ix]))

            if kpList[y_ix] < y_decay: #bande noire haute
                kpList[y_ix] = y_decay
            elif kpList[y_ix] > (sH-y_decay): #idem, basse
                kpList[y_ix] = sH - y_decay
            else:
                kpList[y_ix] -= y_decay
                
            kpList[y_ix] /= scale
            kpList[y_ix] += bbox[1]
            kpList[y_ix] = round(kpList[y_ix], 2)
            #kpList[y_ix] = int(round(kpList[y_ix]))
    
    return list(kpList)