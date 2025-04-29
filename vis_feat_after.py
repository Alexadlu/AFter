import importlib
import linecache
import math
import os
import sys
import time
import numpy as np
env_path = '../AFter'
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale
from pytracking.utils.loading import load_network
from pytracking.tracker import tomp
from pytracking.evaluation import get_dataset
# from get_lasher import get_seqlist

def read_image(image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def FeatureMapVisible(outputs,type):
    outputs = (outputs ** 2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)

    font = cv.FONT_HERSHEY_COMPLEX  # 设置字体
    # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
    # imgzi = cv2.putText(img, "v2t", (20,30), font, 1, (255, 255, 255), 2)
    for j in range(outputs.size(0)):
        am = outputs[j, ...].cpu().numpy()
        am = cv.resize(am, (288, 288))
        am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        
        am=np.stack((am,am,am),axis=0)
        am=np.transpose(am,(1,2,0))  # 这里转换通道为RGB

        am=cv.applyColorMap(am,cv.COLORMAP_JET)

        return am

'''可能有用的序列：234-cycle4_feat glass2
lasher-4men leftunderbasket
vtuavst-pedestrian_123
'''
if __name__=='__main__':
    dataset_name = 'lashertestingset'
    vis_path = '../AFter/pytracking/vis/vis_feat/{}'.format(dataset_name)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('tomp', 'tomp50'))
    params = param_module.parameters()
    tracker = tomp.ToMP(params)
    dataset = get_dataset(dataset_name)
    
    
    for seq in dataset:
        init_info = seq.init_info()
        image_v = read_image(seq.frames_v[0])
        image_t = read_image(seq.frames_i[0])
        out = tracker.initialize(image_v, image_t, init_info)
        frame_num=0
        for frame_path_v, frame_path_i in zip(seq.frames_v[1:], seq.frames_i[1:]):
            frame_num += 1
            image_v = read_image(frame_path_v) # load each frame image
            image_i = read_image(frame_path_i)
            info = seq.frame_info(frame_num)
            im_v,im_t,(test_feat_oral,test_feat) = tracker.track(image_v,image_i, info, vis_weight=False,vis_feat=True)
            im_v = im_v[0].permute(1,2,0)
            im_t = im_t[0].permute(1,2,0)
            
            test_feat = FeatureMapVisible(test_feat,None)
            test_feat_oral = FeatureMapVisible(test_feat_oral,None)
            test_feat = torch.tensor(test_feat)[:,:,[2,1,0]]    # BGR->RGB
            test_feat_oral = torch.tensor(test_feat_oral)[:,:,[2,1,0]]
            
            im_h = im_v.shape[0]
            margin = torch.ones((im_h,20,3))*255       # 图像间距
            imgs_feat = torch.cat([im_v,margin,im_t,margin,test_feat_oral,margin,test_feat],1)
            
            gt = seq.ground_truth_rect[frame_num]
            x,y,w,h = gt
            x,y,w,h = [int(x) for x in gt]
            
            # 画搜索区及对应特征图
            plt.imshow(imgs_feat/255)
            plt.axis(False)
            plt.savefig(os.path.join(vis_path,'{}_feat.jpg'.format(seq.name)),bbox_inches='tight', dpi=200, pad_inches=0.0)
            plt.close()
            
            # 画带gt的图
            image_v = torch.tensor(image_v)
            # image_t = torch.tensor(image_t)
            # imgs_gt = torch.cat([image_v,image_t],1)
            plt.imshow(image_v/255)
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((x,y), w, h, color="red", fill=False, linewidth=2))
            # ax.add_patch(plt.Rectangle((im_h+x,y), w, h, color="red", fill=False, linewidth=2))
            plt.savefig(os.path.join(vis_path,'{}_gt.jpg'.format(seq.name)),bbox_inches='tight', dpi=200, pad_inches=0.0)
            plt.close()
            break
        print('saved ',seq.name)
        # time.sleep(0.3)    
        
    print('over!')
