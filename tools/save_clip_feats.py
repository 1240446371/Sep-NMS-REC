import argparse
import json
import pickle
import os
import random
import os
import sys
import clip
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from utils.constants import EVAL_SPLITS_DICT
import cv2
from PIL import Image
from time import *

def main(args):
    device_cuda=torch.device('cuda',args.gpu_id)
    jit_model, transform = clip.load('ViT-B/16', device=device_cuda,jit=False)
    IMAGE_DIR = '/home/wj/code/MCN_master2/data/images/train2014/'
    dataset_splitby = args.dataset + '_' + args.splitby
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    BOX_FILE_PATH = '/home_data/wj/ref_nms/data/rpn_boxes.pkl'
    SCORE_FILE_PATH = '/home_data/wj/ref_nms/data/rpn_box_scores.pkl'
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    #refer = REFER('/home/wj/code/ref_nms/data', args.dataset, args.split_by)
    #token_path = '/home/wj/code/ref_nms/cache/std_vocab_{}.txt'.format(dataset_splitby)
        
    #noun_tokens = open(token_path)
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005
    img_to_exps = {}
    image_features =[]
    proposals_feats={}
    #token_features ={}
    #dataset_splitby = refdb['dataset_splitby']
    with open(refdb_path) as f:
        refdb = json.load(f)
    for idx, split in enumerate(eval_splits):
        #split = 'train'
        #save_path = '/data1/wj/ref_nms/cache/clip_img_feats_{}_{}.json'.format(dataset_splitby,split)
        #print("savepath%s"%save_path)
        refs = refdb[split]
        print("split%s"%split)
    
    #feats_h5 = osp.join('cache/clipfeats', dataset_splitby)
    #f = h5py.File(feats_h5, 'w')
    #clipfeats_box = f.create_dataset('clipfeats_img', (num_dets, 512), dtype=np.float32)
    #clipfeats_token = f.create_dataset('clipfeats_token', (num_dets, 512), dtype=np.float32)
    
        for ref in refs:
            image_id = ref['image_id']
            if image_id in img_to_exps:
                    img_to_exps[image_id].append((ref['exp_id'], ref['tokens']))
            else:
                    img_to_exps[image_id] = [(ref['exp_id'], ref['tokens'])]
        with open(BOX_FILE_PATH, 'rb') as f:
                img_to_det_box = pickle.load(f)
        with open(SCORE_FILE_PATH, 'rb') as f:
                img_to_det_score = pickle.load(f)
        print("61len img_to_exps.items()%s"%len(img_to_exps.items()))
        k=0
        for image_id, exps in img_to_exps.items():
                
                # Load image feature
                
               
                begin_time = time()
                img_pth=os.path.join(IMAGE_DIR,'COCO_train2014_'+str(image_id).zfill(12)+'.jpg')
                im = cv2.imread(img_pth)
                #print("im%s"%im)
                # RoI-pool positive M-RCNN detections
                det_box = img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
                det_score = img_to_det_score[image_id]  # [300, 81]
                det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])  # [80, 300, 4]
                det_score = np.transpose(det_score[:, 1:], axes=[1, 0])  # [80, 300]
                this_thresh = CONF_THRESH
                positive = det_score > this_thresh  # [80, 300]
                while np.sum(positive) == 0: 
                    this_thresh -= DELTA_CONF
                    positive = det_score > this_thresh  # [80, 300]
                pos_box = torch.tensor(det_box[positive])  # [num_posbox, 4] num is different every image
                #pos_score = torch.tensor(det_score[positive], device=device_cuda)  # [*]
                #cls_num_list = np.sum(positive, axis=1).tolist()  # [80] 80element,the i element is the total pos box num at the i class
                pos_box = pos_box.to(device_cuda)
                #print("pos_box.shape[0]%s"%(pos_box.shape[0]))
                #exp sentence and tokens [(71, [2188, 2071, 950, 1064]), (72, [0, 2071, 1061]), (73, [441, 3974, 1881, 1493, 2500, 660]), (74, [3668]),# (75, [3703, 3668, 660, 540, 3019]), (76, [3668])]        
                for n in range(pos_box.shape[0]):
                    #print("73n%s"%n)
                    #print("74pos_box.shape %s"%pos_box.shape[0])
                    box_array = pos_box[n,:]
                    x1,y1,x2,y2=int(pos_box[n,0]),int(pos_box[n,1]), int(pos_box[n,2])+1,int(pos_box[n,3])+1
                    #print("x1 %s y1 %s x2 %s y2%s"%(x1,y1,x2,y2))
                    im_box = im[y1:y2,x1:x2]
                    im_box=Image.fromarray(im_box)
                    with torch.no_grad():
                        roi_input = transform(im_box).unsqueeze(0).to(device_cuda) 
                        re = clip.tokenize('').to(device_cuda)
                        _,box_feature,_,logits_per_image, per_text = jit_model(roi_input, re)
                        box_feature = torch.squeeze(box_feature).tolist()
                        #print("143boxfeature%s"%(box_feature.shape,))
                        image_features.append({ 'box': box_array.tolist(), 'box_feature': box_feature})
                        
                proposals_feats[image_id] = image_features 
                #print(" 100proposals_feats[image_id]%s"%( proposals_feats[image_id],))
                end_time = time()
                run_time = end_time-begin_time
                k+=1
                print("92 image %s  time is %s num is %s"%(image_id,run_time,k)) 
        save_path = '/data1/wj/ref_nms/cache/clip_img_feats_{}_{}.json'.format(dataset_splitby,split)
        
        with open(save_path, 'w') as f:
            json.dump(proposals_feats, f)
            print('saving img_feats_to {}'.format(save_path))
            
            

            
    # cacaulate score:
 # for imgid in proposals_feats.keys():
  #      for 
     
      
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--splitby', default='unc')
    parser.add_argument('--tid', type=str, default='1019204514')
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
