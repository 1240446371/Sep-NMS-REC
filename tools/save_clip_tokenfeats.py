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
from utils.constants import EVAL_SPLITS_DICT
import cv2
from PIL import Image
from time import *

def main(args):
    device_cuda=torch.device('cuda', args.gpu_id)
    jit_model, transform = clip.load('ViT-B/16', device=device_cuda,jit=False)
    dataset_splitby = args.dataset + '_' + args.splitby
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
  
    #token_path = '/home/wj/code/ref_nms/cache/std_vocab_{}.txt'.format(dataset_splitby)
    token_path = '/home_data/wj/ref_nms/cache/std_vocab_{}.txt'.format(dataset_splitby)    
    noun_tokens = open(token_path)
     
    token_features ={}
    k =0
    for token in noun_tokens:
    #for token in noun_tokens.readlines():
            token=token.strip('\n')
            print("34token%s"%token)
            begin = time()
            k+=1
            img_ = np.zeros((224,224))
            #print("imgshape%s"%(img_.shape,))
            img_=Image.fromarray(img_)
            img_ = transform(img_).unsqueeze(0).to(device_cuda)
            re = clip.tokenize(token).to(device_cuda)
            logit_scale,_,token_feat,_, _ = jit_model(img_, re)
            #print("token_feat%s"%(token_feat.shape,))
            token_features[token] = torch.squeeze(token_feat).tolist()
            #print("156tokenfeature%s"%token_features)
            end = time()
            run_time = end-begin
            print("43 token %s  time is %s num is %s"%(token,run_time,k)) 
    save_path = '/data1/wj/ref_nms/cache/clip_token_feats_{}.json'.format(dataset_splitby)
    print('saving token_feats_to {}'.format(save_path))
    with open(save_path, 'w') as f:
            json.dump( token_features, f)  

  
      
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--splitby', default='unc')
    main(parser.parse_args())
