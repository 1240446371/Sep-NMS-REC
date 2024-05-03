# coding=gbk
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
from lib.refer import REFER
from time import *
import torch.nn.functional as F
from torchvision.ops import nms
def main(args):
    device_cuda=torch.device('cuda',args.gpu_id)
    jit_model, transform = clip.load('ViT-B/16', device=device_cuda,jit=False)
    #IMAGE_DIR = '/home/wj/code/MCN_master2/data/images/train2014/'
    IMAGE_DIR ="/data1/qyx/VQA-X/Images/train2014/"
    dataset_splitby = args.dataset + '_' + args.splitby
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    #BOX_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_boxes.pkl'
    BOX_FILE_PATH = "/home_data/wj/ref_nms/data/rpn_boxes.pkl"
    SCORE_FILE_PATH = "/home_data/wj/ref_nms/data/rpn_box_scores.pkl"
    #SCORE_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_box_scores.pkl'
    #refdb_path = '/home/wj/code/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    #refer = REFER('/home/wj/code/ref_nms/data', args.dataset, args.splitby)
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    refer = REFER('/home_data/wj/ref_nms/data/refer', args.dataset, args.splitby)
    #token_path = '/home/wj/code/ref_nms/cache/std_vocab_{}.txt'.format(dataset_splitby)
    token_feats_path =  "/data1/wj/ref_nms/cache/clip_token_feats_{}.json".format(dataset_splitby)
    #noun_tokens = open(token_path)
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005
   
    #token_features ={}
    #dataset_splitby = refdb['dataset_splitby']
    with open(refdb_path) as f:
        refdb = json.load(f)
    for idx, split in enumerate(eval_splits):
        #split = 'train'
        refs = refdb[split]
        print("split%s"%split)
        img_to_exps = {}
        image_features =[]
        proposals_feats={}
        proposal_dict = {}
        results = {}
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
        num=0
        exp_to_proposals={}  
        total_begin = time()
        for image_id, exps in img_to_exps.items(): # all express
                # Load image feature
                
                
                img_begin = time()
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
                pos_score = torch.tensor(det_score[positive], device=device_cuda)  # [*]
                #cls_num_list = np.sum(positive, axis=1).tolist()  # [80] 80element,the i element is the total pos box num at the i class
                pos_box = pos_box.to(device_cuda)
                cls_num_list = np.sum(positive, axis=1).tolist() 
                #print("pos_box.shape[0]%s"%(pos_box.shape[0]))
                #exp sentence and tokens [(71, [2188, 2071, 950, 1064]), (72, [0, 2071, 1061]), (73, [441, 3974, 1881, 1493, 2500, 660]), (74, [3668]),# (75, [3703, 3668, 660, 540, 3019]), (76, [3668])]        
                begin_time = time()
                for n in range(pos_box.shape[0]):
                    #print("73n%s"%n)
                    #print("74pos_box.shape %s"%pos_box.shape[0])
                    box_array = pos_box[n,:] #4
                    x1,y1,x2,y2=int(pos_box[n,0]),int(pos_box[n,1]), int(pos_box[n,2])+1,int(pos_box[n,3])+1
                    #print("x1 %s y1 %s x2 %s y2%s"%(x1,y1,x2,y2))
                    im_box = im[y1:y2,x1:x2]
                    im_box=Image.fromarray(im_box)
                    with torch.no_grad():
                        roi_input = transform(im_box).unsqueeze(0).to(device_cuda) 
                        re = clip.tokenize('').to(device_cuda)
                        logit_scale,box_feature,_,logits_per_image, per_text = jit_model(roi_input, re)
                        box_feature = torch.squeeze(box_feature).tolist()
                        #print("106boxarrayshape%s"%(box_array.shape,))
                        image_features.append({ 'box': box_array.tolist(), 'box_feature': box_feature})
                #end_time = time()
                #run_time = end_time-begin_time
                #print("run_time%s"%run_time)
                # read token feats:        
                with open(token_feats_path,'r')as fp: # encoding='utf8'
                     token_feats = json.load (fp) 
                     print("load token_feats ok")

                for exp_id,_  in exps:        
                     tokens= refer.sentToTokens[exp_id] # one express
                     # why proposal =0 ?
                     proposals = []
                     for token in tokens:
                          print("token%s"%token)
                          #print("126token_feats.keys()%s"%(token_feats.keys(),))
                          if token not in token_feats.keys():
                              print("token is not in voc%s"%token)
                              continue
                          #token_num+=1
                          #print("70token_num%s"%token_num)
                          rank_score_list=[]
                          rank_score_norm=[]
                          #box_num = pos_box.shape[0]
                          for n in range(pos_box.shape[0]):
                               # caculate  simlarity: read img features and token features
                               #print("138 image_features[n]%s"%image_features[n])
                               #print("139 image_features type%s"%type(image_features)) #list
                               #print("140 image_features type%s"%type(image_features[n])) #dict
                               img_feats = (image_features[n])['box_feature']  # list 
                               img_feats = torch.from_numpy(np.array(img_feats)).to(device_cuda)  # list to tensor
                               text_feats =torch.from_numpy(np.array(token_feats[token])).to(device_cuda)
                               #print("143logit_scale%s"% type(logit_scale))
                               #print("144test_feats%s"% type(text_feats))
                               logits = logit_scale * img_feats @ text_feats.t()  # logit_scale: tesnor
                               #print("127logits%s"%logits)
                               rank_score_list.append(logits)        
                                  
                               #print(" 100proposals_feats[image_id]%s"%( proposals_feats[image_id],))
                              # rank_score_norm = torch.Tensor(rank_score_norm).to(device)
                          # softmax and nms ，cls_num_list？
                          #print("154rank_score_list%s"%(type(rank_score_list)))
                          #rank_score_norm = torch.from_numpy((rank_score_list.cpu().numpy())).to(device_cuda)
                          rank_score_norm =torch.tensor(rank_score_list).to(device_cuda)
                          rank_score_norm = F.softmax(rank_score_norm)
                          min_score = min(rank_score_norm)
                          max_score = max(rank_score_norm)
                          rank_score_norm_list=[]
                          for i in rank_score_norm:
                              score_norm=float(format((i-min_score)/(max_score-min_score),'.4f'))
                              rank_score_norm_list.append(score_norm)
                          rank_score_norm_list =torch.tensor(rank_score_norm_list).to(device_cuda)    
                          #rank_score_norm =torch.from_numpy(rank_score_list)
                          #rank_score_norm =np.array(rank_score_list.to(device_cpu))
                          #print("158rank_score_list%s"%(type(rank_score_norm)))
                               # too samll 
                          
                          print("157rank_score_norm%s"%(rank_score_norm_list,))
                          rank_score_norm_list = torch.split(rank_score_norm_list, cls_num_list, dim=0)  # tuple
                          #print("58rank_scorelist shape%s"%type(rank_score_norm))
                          pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
                          pos_score_list = torch.split(pos_score, cls_num_list, dim=0)  
                      # Combine score and do NMS category-wise
                           # place before 
                          cls_idx = 0
                          for cls_rank_score, cls_pos_box, cls_pos_score in zip(rank_score_norm_list, pos_box_list, pos_score_list):
                               cls_idx += 1
                           # No positive box under this category
                               if cls_rank_score.size(0) == 0:
                                   continue
                               final_score = cls_rank_score * cls_pos_score
                               keep = nms(cls_pos_box, final_score, iou_threshold=0.3)  # according to the final score to rank box, and caculate the iou to filter box
                               cls_kept_box = cls_pos_box[keep]
                               cls_kept_score = final_score[keep]
                               for box, score in zip(cls_kept_box, cls_kept_score):
                                   proposals.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})  # one exp_id , one proposal init init after exp_id
                           #assert cls_idx == 80
                     exp_to_proposals[exp_id] = proposals  #  save all exp ,init at the begin  save in one exp 
                img_end = time()
                img_time = img_end - img_begin
                num += 1
                print("num  %s  img time is %s"%(num,img_time))   
        #save_path = '/home/wj/code/ref_nms/cache/clip_img_feats_{}_{}.json'.format(dataset_splitby,split)
         # save in one slpit 
        proposal_dict[split] = exp_to_proposals  # save in one slpit 
        pro_save_path = '/data1/wj/ref_nms/cache/clipproposals1_{}_{}.pkl'.format( dataset_splitby,split)
        #pro_save_path = '/home/wj/code/ref_nms/cache/clipproposals_{}_{}.json'.format( dataset_splitby,split)
        total_end = time()
        total_time = total_end - total_begin 
        print('saving proposals to {}...'.format(pro_save_path))
        with open(pro_save_path, 'wb') as f:
            pickle.dump(proposal_dict, f)
            print("save ok and total time is %s"%(total_time))
            #json.dump(proposal_dict, f)
        """with open(save_path, 'w') as f:
            json.dump(proposals_feats, f)
            print('saving img_feats_to {}'.format(save_path))   #add result [split] dict"""
            
            

            
    # cacaulate score:
 # for imgid in proposals_feats.keys():
  #      for 
     
      
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--splitby', default='unc')
    parser.add_argument('--tid', type=str, default='1019204514')
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
