import pickle
import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

from utils.misc import calculate_iou, mrcn_crop_pool_layer  #

from lib.refer import REFER
from transformers import AutoTokenizer, BertModel
from transformers import logging

logging.set_verbosity_warning()


__all__ = ['DetBoxDataset', 'DetEvalLoader', 'DetEvalTopLoader']


class DetBoxDataset(Dataset):

    #HEAD_FEAT_DIR = '/home_data/wj/ref_nms/data/head_feats'
    #HEAD_FEAT_DIR= '/home/wj/code/MAttNet_master/cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime/'
    #HEAD_FEAT_DIR= '/home/wj/code/MAttNet_master/cache/feats/refcoco+_unc/mrcn/res101_coco_minus_refer_notime/'
    #HEAD_FEAT_DIR= '/home/wj/code/MAttNet_master/cache/feats/refcocog_umd/mrcn/res101_coco_minus_refer_notime/'
    HEAD_FEAT_DIR="/home_data/wj/matt/cache/feats/refcoco+_unc/mrcn/res101_coco_minus_refer_notime/"  # coco+ 18
    #BOX_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_boxes.pkl' 
    #SCORE_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_box_scores.pkl'
    BOX_FILE_PATH ="/home_data/wj/ref_nms/data/rpn_boxes.pkl"
    SCORE_FILE_PATH ="/home_data/wj/ref_nms/data/rpn_box_scores.pkl"
    
     
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005
    #tokenize_model_name = 'distillery-base-uncashed-finetuned-still-2-english'
    bert_model_name = 'bert-base-uncased'



    def __init__(self, tokenizer,textmodel,dataset,split_by,Images,refdb, ctxdb, split, roi_per_img):
        Dataset.__init__(self)
        self.refs = refdb[split]
        self.dataset_splitby = refdb['dataset_splitby']
        print("31self.dataset_splitby%s"%self.dataset_splitby)
        self.exp_to_ctx = ctxdb[split]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        #self.idx_to_glove = np.load('/home/wj/code/ref_nms/cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.idx_to_glove = np.load('/home_data/wj/ref_nms/cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10
        self.pad_feat = np.zeros(300, dtype=np.float32)
        # Number of samples to draw from one image
        self.roi_per_img = roi_per_img
        
        #self.refer = REFER('/home/wj/code/ref_nms/data/', dataset, split_by)
        self.refer = REFER('/home_data/wj/ref_nms/data/refer/', dataset, split_by)
        #self.tokenize_model=AutoModelForSequenceClassification.from_pretrained(self.tokenize_model_name)
        #self.tokenizer=AutoTokenizer.from_pretrained(self.tokenize_model_name)
        logging.set_verbosity_warning()
        
        #self.tokenizer=AutoTokenizer.from_pretrained(self.bert_model_name)
        #self.textmodel = BertModel.from_pretrained(self.bert_model_name)
        
        self.tokenizer=tokenizer
        self.textmodel = textmodel
        
        self.Images = Images
 
            #device_cuda=torch.device('cuda',1)
    



    def __getitem__(self, idx):
        """
        Returns:
            roi_feats: [R, 1024, 7, 7]
            roi_labels: [R]
            word_feats: [S, 300]
            sent_len: [0]
        """
        # Index refer object
        # obtain sentence according to sent id or ref_id
        # refer 
        ref = self.refs[idx] 
        image_id = ref['image_id']
        gt_box = ref['bbox'] # xyxy
        exp_id = ref['exp_id'] # xyxy
        # obtain sent
        sent = self.refer.Sents[exp_id]['sent']
        #print("79 sent %s"%sent)
        #print("80 exp_id %s"%exp_id)
        # obtain sentfeats from bert
        sent_feats,bert_word_feats, sent_len= self.build_sent_feats(ref['tokens'],sent)
        #sent_feats =sent_feats.detach()
        
        ctx_boxes = [c['box'] for c in self.exp_to_ctx[str(exp_id)]['ctx']]# xyxy
        #target_list = [gt_box] + ctx_boxes
        #pos_rois, neg_rois = self.get_labeled_rois(image_id, target_list)
        gt_pos_rois,ctx_pos_rois, neg_rois = self.get_labeled_rois(image_id,gt_box,ctx_boxes)
        # Build word features
        #print("gt_pos_rois%s,ctx_pos_rois%s, neg_rois%s"%(len(gt_pos_rois),len(ctx_pos_rois),len(neg_rois)))
        #print("64ref['tokens']%s"%(ref['tokens'],))
        #word_feats, sent_len = self.build_word_feats(ref['tokens'])
        # Load image feature
        #print("60path%s"%(os.path.join(self.HEAD_FEAT_DIR,'{}.h5'.format(image_id))))
        #image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, self.dataset_splitby, '{}.h5'.format(image_id)), 'r')
        image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
        scale = image_h5['im_info'][0, 2]
        image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
        # Sample ROIs
        gt_num = min(len(gt_pos_rois), 4)
        ctx_pos_num = min(len(ctx_pos_rois), 21-gt_num)
        neg_num = min(len(neg_rois), self.roi_per_img- gt_num - ctx_pos_num)
        ctx_pos_num = min(len(ctx_pos_rois),self.roi_per_img- gt_num -neg_num )
        gt_num = min(len(gt_pos_rois),self.roi_per_img- ctx_pos_num -neg_num )
        #print(" gt_num %s ctx_pos_num%s"% (gt_num,ctx_pos_num))
        #print("neg_num%s"%neg_num)
        sampled_gt_pos = random.sample(gt_pos_rois, gt_num)
        sampled_ctx_pos = random.sample(ctx_pos_rois, ctx_pos_num)
        sampled_neg = random.sample(neg_rois, neg_num)  
        #print("83sampled_gt_pos%s"%sampled_gt_pos)
        pos_labels = torch.ones(len(sampled_ctx_pos)+len(sampled_gt_pos), dtype=torch.float)
        neg_labels = torch.zeros(len(sampled_neg), dtype=torch.float)
        roi_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [R]
        
        pos_ctx_labels = torch.ones(len(sampled_ctx_pos), dtype=torch.float)
        gt_labels = torch.ones(len(sampled_gt_pos), dtype=torch.float)
        ctx_neg_labels = torch.zeros(len(sampled_ctx_pos)+len(sampled_neg), dtype=torch.float)
        #labels_gt = torch.cat([gt_labels, ctx_neg_labels], dim=0)  # [R]
        labels_gt_neg = torch.cat([gt_labels,ctx_neg_labels], dim=0) #  [R]
        # Extract head features
        #sampled_roi = torch.tensor(sampled_pos + sampled_neg)    # [R, 4]
        sampled_roi = torch.tensor(sampled_gt_pos +sampled_ctx_pos+ sampled_neg) 
        sampled_roi.mul_(scale)
        roi_feats = mrcn_crop_pool_layer(image_feat, sampled_roi)
        #print("roifeat shape %s "%(roi_feats.shape,))  # ([32, 1024, 7, 7])
        #print("80van roi_feats%s gt_num%s,ctx_pos_num%s, neg_num%s,roi_labels%s ,word_feats%s sent_len%s "%(roi_feats.shape, gt_num,ctx_pos_num, neg_num, roi_labels.shape, word_feats.shape, sent_len))
        lfeats =  self.compute_lfeats(image_id,sampled_gt_pos+sampled_ctx_pos+sampled_neg)
        #print("126.....loader bert %s"%(sent_feats.shape,))
        sent_feats = sent_feats.detach()
        bert_word_feats = bert_word_feats.detach()
        #requires_grad
        #print("143 roi_labels%s%s"%(roi_labels.requires_grad,roi_labels.device))
        #print("144labels_gt_neg%s%s"%(labels_gt_neg.requires_grad,labels_gt_neg.device,))
        #print("145word_feats%s%s"%(word_feats.requires_grad,word_feats.device,))
        #print("146sent_feats%s%s"%(sent_feats.requires_grad,sent_feats.device,))
        #print("147lfeats%s"%(lfeats.requires_grad,))
        return roi_feats, roi_labels,labels_gt_neg, bert_word_feats, sent_len,sent_feats,lfeats  # bert word

    def __len__(self):
        return len(self.refs)

# first score >class confidence 0.05, then the num =roiperimg , then iou>0.5 is pos then is neg

        
    def get_labeled_rois(self, image_id, gt_box,ctx_boxes):
        boxes = self.img_to_det_box[image_id].reshape(-1, 81, 4)
        scores = self.img_to_det_score[image_id]
        boxes = boxes[:, 1:]  # [*, 80, 4]
        scores = scores[:, 1:]  # [*, 80]
        # boxes = boxes.reshape(-1, 4)
        # scores = scores.reshape(-1)
        # top_idx = np.argsort(scores)[-self.TOP_N:]
        this_thresh = self.CONF_THRESH
        positive = scores > this_thresh
        while np.sum(positive) < self.roi_per_img:
            this_thresh -= self.DELTA_CONF
            positive = scores > this_thresh
        ctx_pos_rois = []
        gt_pos_rois = []
        neg_rois = []

        for box in boxes[positive]:
            tag=0
            if calculate_iou(box, gt_box) >= 0.5:
                gt_pos_rois.append(box)
                tag=1
                #break
            else:
                for t in ctx_boxes:
                    if calculate_iou(box, t) >= 0.5:
                       ctx_pos_rois.append(box)
                       tag=1
                       break
            if tag==0:
                neg_rois.append(box)
        
        return gt_pos_rois,ctx_pos_rois, neg_rois

    def build_word_feats(self, tokens):
        word_feats = [self.idx_to_glove[wd_idx] for wd_idx in tokens]
        #print("van119wordfeats%s"%word_feats)
        #print("van120len(word_feats)%s"%len(word_feats))
        word_feats += [self.pad_feat] * max(self.max_sent_len - len(word_feats), 0)  # pad 0 to 30 cocog /20coco
        #print("van122 +len(word_feats)%s"%len(word_feats))
        word_feats = torch.tensor(word_feats[:self.max_sent_len])  # [S, 300]
        #print("van124 word_feats%s"%(word_feats.shape,))
        print("197 len tokens %s"%len(tokens))
        return word_feats, min(len(tokens), self.max_sent_len)
        
    def build_sent_feats(self,tokens, sent):
    
        sent_tokenize = self.tokenizer(sent,padding='max_length',max_length= self.max_sent_len+1,truncation=True,return_tensors="pt")
        
        #print("179sent_tokenize %s"%(sent_tokenize))
        #sent_tokenize =sent_tokenize.detach()
        #all_encoder_layers = textmodel(**sent_tokenize,output_hidden_states=True)
        output = self.textmodel(**sent_tokenize,output_hidden_states=True)
        
        #output =output.detach()
        ## Sentence feature at the first position [cls]
        #print("185 output%s"%output) 
        # mean hidden_state
        hidden_states=output[2]
        #print("186 hidden_states %s"%(hidden_states,))
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        
        last_hidden_state = output[0]
        word_embedding = last_hidden_state[:, 1:, :]  #[b,len,512]
        #print("219.......word_embedding%s"%(word_embedding.shape,))
        #print("220.......sentlen%s"%(min(len(tokens), self.max_sent_len)))
           
        #Sentence feature at the first position [cls]
        #raw_flang = (all_encoder_layers[-1][0,:] + all_encoder_layers[-2][0,:]+ all_encoder_layers[-3][0,:] + all_encoder_layers[-4][0,:])/4
        """
        raw_flang =(all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
                 + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
        print("185 raw_flang %s"%raw_flang)  
        
        ## fix bert during training
        # raw_flang = raw_flang.detach()
        """
        return sentence_embedding,word_embedding, min(len(tokens), self.max_sent_len)
        
    def compute_lfeats(self, image_id,boxlist):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(boxlist), 5), dtype=np.float32)
        image = self.Images[image_id]
        ih, iw = image['height'], image['width']
        for ix, box in enumerate(boxlist):
          x1, y1, x2, y2 = box
          x=x1
          y=y1
          w=x2 - x1
          h=y2 - y1
          #print("176w  %s h %s"%(w,h))
          lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
        #print("173lfeat%s"%(lfeats.shape,))
        return lfeats


class DetBoxDatasetNoCtx(Dataset):

    HEAD_FEAT_DIR = '/home_data/wj/ref_nms/cache/head_feats/matt-mrcn'
    BOX_FILE_PATH = '/home_data/wj/ref_nms/cache/rpn_boxes.pkl'
    SCORE_FILE_PATH = '/home_data/wj/ref_nms/cache/rpn_box_scores.pkl'
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005

    def __init__(self, refdb, split, roi_per_img):
        Dataset.__init__(self)
        self.refs = refdb[split]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('/home_data/wj/ref_nms/cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10
        self.pad_feat = np.zeros(300, dtype=np.float32)
        # Number of samples to draw from one image
        self.roi_per_img = roi_per_img

    def __getitem__(self, idx):
        """

        Returns:
            roi_feats: [R, 1024, 7, 7]
            roi_labels: [R]
            word_feats: [S, 300]
            sent_len: [0]

        """
        # Index refer object
        ref = self.refs[idx]
        image_id = ref['image_id']
        gt_box = ref['bbox']
        pos_rois, neg_rois = self.get_labeled_rois(image_id, gt_box)
        # Build word features
        word_feats, sent_len = self.build_word_feats(ref['tokens'])
        # Load image feature
        image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
        scale = image_h5['im_info'][0, 2]
        image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
        # Sample ROIs
        pos_num = min(len(pos_rois), self.roi_per_img // 2)
        neg_num = min(len(neg_rois), self.roi_per_img - pos_num)
        pos_num = self.roi_per_img - neg_num
        sampled_pos = random.sample(pos_rois, pos_num)
        sampled_neg = random.sample(neg_rois, neg_num)
        pos_labels = torch.ones(len(sampled_pos), dtype=torch.float)
        neg_labels = torch.zeros(len(sampled_neg), dtype=torch.float)
        roi_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [R]
        # Extract head features
        sampled_roi = torch.tensor(sampled_pos + sampled_neg)    # [R, 4]
        sampled_roi.mul_(scale)
        roi_feats = mrcn_crop_pool_layer(image_feat, sampled_roi)
        return roi_feats, roi_labels, word_feats, sent_len

    def __len__(self):
        return len(self.refs)

    def get_labeled_rois(self, image_id, gt_box):
        boxes = self.img_to_det_box[image_id].reshape(-1, 81, 4)
        scores = self.img_to_det_score[image_id]
        boxes = boxes[:, 1:]  # [*, 80, 4]
        scores = scores[:, 1:]  # [*, 80]
        this_thresh = self.CONF_THRESH
        positive = scores > this_thresh
        while np.sum(positive) < self.roi_per_img:
            this_thresh -= self.DELTA_CONF
            positive = scores > this_thresh
        pos_rois = []
        neg_rois = []
        # for box in boxes[top_idx]:
        for box in boxes[positive]:
            if calculate_iou(box, gt_box) >= 0.5:
                pos_rois.append(box)
            else:
                neg_rois.append(box)
        return pos_rois, neg_rois

    def build_word_feats(self, tokens):
        word_feats = [self.idx_to_glove[wd_idx] for wd_idx in tokens]
        word_feats += [self.pad_feat] * max(self.max_sent_len - len(word_feats), 0)
        word_feats = torch.tensor(word_feats[:self.max_sent_len])  # [S, 300]
        return word_feats, min(len(tokens), self.max_sent_len)



class DetEvalLoader:

    BOX_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_boxes.pkl'
    SCORE_FILE_PATH = '/home/wj/code/ref_nms/data/rpn_box_scores.pkl'
    #IMG_FEAT_DIR = '/home/wj/code/MAttNet_master/cache/feats/refcocog_umd/mrcn/res101_coco_minus_refer_notime/'
    IMG_FEAT_DIR = '/home/wj/code/MAttNet_master/cache/feats/refcoco+_unc/mrcn/res101_coco_minus_refer_notime/'
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005

    def __init__(self, tokenizer,textmodel,Images,refdb, split='val', gpu_id=0):
        self.dataset_splitby = refdb['dataset_splitby']
        self.refs = refdb[split]
        self.img_to_exps = {}
        for ref in self.refs:
            image_id = ref['image_id']
            if image_id in self.img_to_exps:
                self.img_to_exps[image_id].append((ref['exp_id'], ref['tokens']))
            else:
                self.img_to_exps[image_id] = [(ref['exp_id'], ref['tokens'])]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('/home/wj/code/ref_nms/cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.device = torch.device('cuda', gpu_id)
        self.Images = Images
        #add refer loader
                
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10        
        self.tokenizer=tokenizer
        self.textmodel = textmodel
        self.dataset_ = self.dataset_splitby.split('_')[0]
        self.split_by = self.dataset_splitby.split('_')[1]
        self.refer = REFER('/home/wj/code/ref_nms/data/', self.dataset_, self.split_by)


    def __iter__(self):
        # Fetch ref info
        print("305.......")
        for image_id, exps in self.img_to_exps.items():
            # Load image feature
            #image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, self.dataset_splitby, '{}.h5'.format(image_id)), 'r')
            image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
            scale = image_h5['im_info'][0, 2]
            image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
            #print("312 .......")
            # RoI-pool positive M-RCNN detections
            det_box = self.img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
            det_score = self.img_to_det_score[image_id]  # [300, 81]
            det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])  # [80, 300, 4]
            det_score = np.transpose(det_score[:, 1:], axes=[1, 0])  # [80, 300]
            this_thresh = self.CONF_THRESH
            positive = det_score > this_thresh  # [80, 300]  # 
            #print("367.....posshape%s pos%s"%(positive.shape,positive,))
            while np.sum(positive) == 0:
                this_thresh -= self.DELTA_CONF
                positive = det_score > this_thresh  # [80, 300]
            pos_box = torch.tensor(det_box[positive])  # [*, 4]
            # add loc 
            
            lfeats = np.empty((len(pos_box), 5), dtype=np.float32)
            image = self.Images[image_id]
            ih, iw = image['height'], image['width']
            for ix, box in enumerate(pos_box):
              x1, y1, x2, y2 = box
              x=x1
              y=y1
              w=x2 - x1
              h=y2 - y1
              lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
            #print("336.......")
            
            pos_score = torch.tensor(det_score[positive], device=self.device)  # [*]   all pos box
            #print("407.....len%s score%s"%(pos_score.shape,pos_score,))  # 156????
            cls_num_list = np.sum(positive, axis=1).tolist()  # [80] for one cat, how many box is pos
            #print("257cls_num_list%s"%(cls_num_list,))#[15,83,0,0,0....] add all=shape of pos score
            #print("258lencls_num_list%s"%len(cls_num_list))
            pos_feat = mrcn_crop_pool_layer(image_feat, pos_box * scale)  # [*, 1024, 7, 7]
            pos_feat = pos_feat.to(self.device).unsqueeze(0)  # [1, *, 1024, 7, 7]
            pos_box = pos_box.to(self.device)
            lfeats = torch.tensor(lfeats).to(self.device).unsqueeze(0)
            for exp_id, tokens in exps:
                # Load word feature
                assert isinstance(tokens, list)
                #print("263tokens %s"%tokens)  #[2688, 540, 2071]
                sent_feat = torch.tensor(self.idx_to_glove[tokens], device=self.device)
                #print("265sent_feats.shape%s"%(sent_feat.shape,))
                sent_feat = sent_feat.unsqueeze(0)  # [1, *, 300]
                sent = self.refer.Sents[exp_id]['sent']
                bert_feats = self.build_sent_feats(sent)
                bert_feats =bert_feats.unsqueeze(0)
                bert_feats= bert_feats.to(self.device)
                yield exp_id,lfeats, pos_feat, sent_feat,bert_feats, pos_box, pos_score, cls_num_list

    def __len__(self):
        return len(self.refs)
        
    def build_sent_feats(self, sent):
    
        sent_tokenize = self.tokenizer(sent,padding='max_length',max_length= self.max_sent_len,truncation=True,return_tensors="pt")

        #all_encoder_layers = textmodel(**sent_tokenize,output_hidden_states=True)
        output = self.textmodel(**sent_tokenize,output_hidden_states=True)
        hidden_states=output[2]
        #print("186 hidden_states %s"%(hidden_states,))     
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()   
        #Sentence feature at the first position [cls]
        #raw_flang = (all_encoder_layers[-1][0,:] + all_encoder_layers[-2][0,:]+ all_encoder_layers[-3][0,:] + all_encoder_layers[-4][0,:])/4
        """
        raw_flang =(all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
                 + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
        print("185 raw_flang %s"%raw_flang)  
        
        ## fix bert during training
        # raw_flang = raw_flang.detach()
        """
        return sentence_embedding


class DetEvalTopLoader:

    BOX_FILE_PATH = '/home_data/wj/ref_nms/cache/rpn_boxes.pkl'
    SCORE_FILE_PATH = '/home_data/wj/ref_nms/cache/rpn_box_scores.pkl'
    IMG_FEAT_DIR = 'cache/head_feats/matt-mrcn'

    def __init__(self, refdb, split='val', gpu_id=0, top_N=200):
        self.refs = refdb[split]
        self.img_to_exps = {}
        for ref in self.refs:
            image_id = ref['image_id']
            if image_id in self.img_to_exps:
                self.img_to_exps[image_id].append((ref['exp_id'], ref['tokens']))
            else:
                self.img_to_exps[image_id] = [(ref['exp_id'], ref['tokens'])]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('/home_data/wj/ref_nms/cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.device = torch.device('cuda', gpu_id)
        self.top_N = top_N

    def __iter__(self):
        # Fetch ref info
        for image_id, exps in self.img_to_exps.items():
            # Load image feature
            image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
            scale = image_h5['im_info'][0, 2]
            image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
            # RoI-pool positive M-RCNN detections
            det_box = self.img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
            det_score = self.img_to_det_score[image_id]                 # [300, 81]
            det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])      # [80, 300, 4]
            det_score = np.transpose(det_score[:, 1:], axes=[1, 0])     # [80, 300]

            this_thresh = np.sort(det_score, axis=None)[-self.top_N]
            positive = det_score >= this_thresh        # [80, 300]
            pos_box = torch.tensor(det_box[positive])  # [*, 4]
            pos_score = torch.tensor(det_score[positive], device=self.device)  # [*]
            cls_num_list = np.sum(positive, axis=1).tolist()                   # [80]

            pos_feat = mrcn_crop_pool_layer(image_feat, pos_box * scale)  # [*, 1024, 7, 7]
            pos_feat = pos_feat.to(self.device).unsqueeze(0)  # [1, *, 1024, 7, 7]
            pos_box = pos_box.to(self.device)
            for exp_id, tokens in exps:
                # Load word feature
                assert isinstance(tokens, list)
                sent_feat = torch.tensor(self.idx_to_glove[tokens], device=self.device)
                sent_feat = sent_feat.unsqueeze(0)  # [1, *, 300]
                yield exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list

    def __len__(self):
        return len(self.refs)


def _test():
    import json
    from tqdm import tqdm
    refdb = json.load(open('/home_data/wj/ref_nms/cache/refdb_refcoco_unc_nopos.json', 'r'))
    ctxdb = json.load(open('/home_data/wj/ref_nms/cache/ctxdb_refcoco_unc.json', 'r'))
    dataset = DetBoxDataset(refdb, ctxdb, 'train')
    neg_num, pos_num, total_num = [], [], []
    for pos_rois, neg_rois in tqdm(dataset, ascii=True):
        neg_num.append(len(neg_rois))
        pos_num.append(len(pos_rois))
        total_num.append(len(pos_rois) + len(neg_rois))
    print('neg min: {}, neg max: {}, neg mean: {}'.format(min(neg_num), max(neg_num), sum(neg_num) / len(neg_num)))
    print('pos min: {}, pos max: {}, pos mean: {}'.format(min(pos_num), max(pos_num), sum(pos_num) / len(pos_num)))
    print('total min: {}, total max: {}, total mean: {}'.format(min(total_num), max(total_num), sum(total_num) / len(total_num)))


if __name__ == '__main__': _test()
