import json
import pickle
import argparse
from multiprocessing import Pool
import os
import sys
import os.path as osp
root_path = os.path.abspath(__file__) 
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import nms
from torch.nn.utils.rnn import pack_padded_sequence
#import clip
#from lib.predictor import AttVanillaPredictorV2
#from lib.predictort1_ls_3cls import AttVanillaPredictorV2
#from lib.vanilla_utils1 import DetEvalLoader
from lib.my_bert_vanilla_utils import DetEvalLoader
#from lib.mypredictor import AttVanillaPredictorV2
#from lib.my_nce_predictor import AttVanillaPredictorV2
#from lib.my_qkv_predictor import AttVanillaPredictorV2
from utils.constants import EVAL_SPLITS_DICT
#from lib.my_bert_predictor import AttVanillaPredictorV2
from lib.my_bertall_predictor import AttVanillaSigadPredictorV2
from transformers import AutoTokenizer, BertModel
import pandas as pd

# load rpn, >0.05 class confidence; then caculate the relate score, combine the class confidence, obtain the threhold
def rank_proposals(position, gpu_id, tid, refdb_path, split, m,dataset_splitby ):
    # Load refdb
    with open(refdb_path) as f:
        refdb = json.load(f)
        #print("27load json ")
    dataset_ = refdb['dataset_splitby'].split('_')[0]
    #print("29 load dataset_ ")
    # Load pre-trained model
   # device = torch.device('cuda', gpu_id)
    device=torch.device('cuda',gpu_id)

    #with open('/home/wj/code/ref_nms/output/neg045_32yb004_{}_{}_{}.json'.format(m, dataset_, tid), 'r') as f:
    
    #json_pth ="/home/wj/code/ref_nms/output/08yb_att_vanilla_refcoco+_0714160018.json"
    #json_pth ="/home/wj/code/ref_nms/output/08yb_att_vanilla_refcocog_0720222934.json"
    #json_pth ="/home_data/wj/ref_nms/output/refnce_t1006_refcoco_unc.json"
    json_pth ="/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_0913191409.json"
    
 
    #with open(json_pth.format(m, dataset_, tid), 'r') as f:
    with open(json_pth,'r') as f:
        #print("jsonpath%s"%f)
        model_info = json.load(f) 
        #print("45jsonpath%s"%f)
    predictor = AttVanillaSigadPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
                                      rank_dropout_p=model_info['config']['RANK_DROPOUT_P'])
    #print("48predictor")
    #model_path = '/home/wj/code/ref_nms/output/{}_yb008_{}_{}_b.pth'.format(m, dataset_, tid)
    #model_path = '/home/wj/code/ref_nms/output/att_vanilla_yb008_refcoco+_0110152420_b.pth'
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb008ckpt_refcoco+_0110152420_2.pth" # 3,4,5
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb004_refcocog_0112101456_b.pth"
    #model_path ='/home/wj/code/ref_nms/output/att_vanilla_yb008_refcoco+_0112095046_b.pth'
    #model_path ='/home/wj/code/ref_nms/output/neg045_32att_vanilla_yb004_refcocog_0318182906_b.pth'
    #/home/wj/code/ref_nms/output/neg045_24yb004_att_vanilla_refcocog_0318142357.json
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb006ckpt_refcoco+_5.pth"
    #model_path  = "/home/wj/code/ref_nms/output/05refyb_att_vanilla_refcoco+_0624154853_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/048refyb_att_vanilla_refcoco+_0625104037_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/048refyb_att_vanilla_refcoco_0626110852_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/03yb_att_vanilla_refcocog_0713144804_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/att_vanilla_refcoco+_1019213349_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/gt_att_vanilla_refcoco+_0725160358_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/ls_refnce_t1006_refcoco+_0815120712_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/ref_loc_nce_t1006_refcocog_0824233302_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/my_att_vanilla_refcocog_0827214414_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/my_nce_att_vanilla_refcocog_0827224102_b.pth"
    #model_path  = "/home_data/wj/ref_nms/output/my_sip_att_vanilla_refcoco_unc_0829102410_b.pth"
    #model_path  = "/home_data/wj/ref_nms/output/my_sipsep_att_vanilla_refcoco_unc_0830001649_b.pth"
    #model_path  = "/home_data/wj/ref_nms/output/my_sipqkvcat_att_vanilla_refcoco_unc_0903150212_b.pth"
    #model_path = "/home_data/wj/ref_nms/output/my_sipqkad_gtctxrefcoco+_1028133418_b.pth"
    #model_path = "/home_data/wj/ref_nms/output/my_bert_attvanilla_refcoco+_unc_1031100427_b.pth"
    model_path = "/home_data/wj/ref_nms/output/my_bertall_qkad_sigad_refcoco+_unc_1113160151_b.pth"
       
    predictor.load_state_dict(torch.load(model_path,map_location=device))
    #print("69predictorload")
    predictor.to(device)
    predictor.eval()

    #exp_to_proposals = {}
    exp_to_proposals_gt1={}
    exp_to_proposals_gt2={}
    exp_to_proposals_gt3={}

  
    
    ###### add images info for loc
    

    data_json =  osp.join('/home/wj/code/MAttNet_master/cache/prepro',dataset_splitby, 'data.json') 
    #ctxdb_path ="/home/wj/code/ref_nms/cache/ybclip_ctxdb_ann_04sen{}.json".format(dataset_splitby)
    #ctxdb_path = "/home/wj/code/ref_nms/cache/ybclip_ctxdb_ann_08senrefcoco_unc.json"
 
    info = json.load(open(data_json))  
    images = info['images'] 
    Images = {image['image_id']: image for image in images}
    
    bert_model_name = 'bert-base-uncased'
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    with torch.no_grad():
      textmodel = BertModel.from_pretrained(bert_model_name)
      textmodel.eval()
    loader = DetEvalLoader(tokenizer,textmodel,Images,refdb, split, gpu_id)
    #print("77 loader ") 
    tqdm_loader = tqdm(loader, desc='scoring {}'.format(split), ascii=True, position=position)
    t_num=0
    #for exp_id,lfeats, pos_feat, sent_feat, bert_feats,pos_box, pos_score, cls_num_list in tqdm_loader:
    for exp_id,pos_box, pos_score, pos_feat, sent_len, bert_token_feats,bert_sent_feats, lfeats, cls_num_list in tqdm_loader:
 
        # Compute rank score
        #t_num+=1
        #print("49t_num%s"%t_num)
        #packed_sent_feats = pack_padded_sequence(sent_feats, torch.tensor([sent_feat.size(1)]),enforce_sorted=False, batch_first=True)  # save first sequence, then second sequence
        #print("112 pos_feat %s,bert_feats %s,lfeats %s"%(pos_feat.shape,bert_feats.shape,lfeats.shape,))
        #print("122........pos_box %s  pos_score %s  pos_feat %s  sent_len %s  bert_token_feats %s bert_sent_feats %s lfeats %s"%(pos_box.device, pos_score.device , pos_feat.device, sent_len.device ,bert_token_feats.device ,bert_sent_feats.device, lfeats.device,))
        
        with torch.no_grad():         
            gt_score,rank_score = predictor(pos_feat, sent_len,bert_token_feats,bert_sent_feats,lfeats)  # [1, n_box]
            #print("126gtscore%srank_score%s"%(gt_score.shape,rank_score.shape,))        
        #rank_score = torch.sigmoid(rank_score[0])#[n_box]
        #print("128.........")
        #rank_score_list = torch.split(rank_score[0], cls_num_list, dim=0)        #80
        #print("130.........")
      
        #print("posscore %s"%(pos_score.shape,)) # nobx
 
        #print("posbox%s"%(pos_box.shape,))  # nbox 4
        gt_score1 = gt_score[0]
        zeros = torch.zeros_like(gt_score1)

        gt_score1 = torch.where(gt_score1>0.01,gt_score1,zeros) # 0.01 ->0 all
    
        gt_score_list = torch.split(gt_score1, cls_num_list, dim=0)  # tuple    
     
        #gt_fx3_list = torch.split(gt_score_fx3, cls_num_list, dim=0)  # tuple      
     
  
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)        
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)    # len()=80

       
        # Combine score and do NMS category-wise

   
        proposals_gt1=[]
     
        proposals_gt2=[]
        proposals_gt3=[]

      
        cls_idx = 0
        for cls_gt1,cls_pos_box, cls_pos_score in zip(gt_score_list,pos_box_list, pos_score_list):
            
            cls_idx += 1
            if cls_gt1.size(0) == 0 : # size(0)some is 0, and some is 16 or other
                continue
            # gt cls
            if split=='val':
                #print("168val........")
                final_score_gt1 = cls_gt1 * (cls_pos_score)**(1/2)
                keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt1 = cls_pos_box[keepgt1]
                cls_kept_score_gt1 = final_score_gt1[keepgt1]
                for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                    proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
                
                
                final_score_gt2 = cls_gt1 * (cls_pos_score)**(1/3)
                keepgt2 = nms(cls_pos_box, final_score_gt2, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt2 = cls_pos_box[keepgt2]
                cls_kept_score_gt2 = final_score_gt2[keepgt2]
                for box2, score2 in zip(cls_kept_box_gt2, cls_kept_score_gt2):
                    proposals_gt2.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})
            
            if split=='testA':
                #print("187.....testA")
               
                final_score_gt1 = ((cls_gt1)**(1/2)) * cls_pos_score
                keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt1 = cls_pos_box[keepgt1]
                cls_kept_score_gt1 = final_score_gt1[keepgt1]
                for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                    proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
                    
                              
                final_score_gt2 = ((cls_gt1)**(1/3)) * cls_pos_score
                keepgt2 = nms(cls_pos_box, final_score_gt2, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt2 = cls_pos_box[keepgt2]
                cls_kept_score_gt2 = final_score_gt2[keepgt2]
                for box2, score2 in zip(cls_kept_box_gt2, cls_kept_score_gt2):
                    proposals_gt2.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})     
                    
            
            if split=='testB':    
                #print("187.....testB")   
                final_score_gt1 = cls_gt1 * (cls_pos_score)**(1/2)
                keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt1 = cls_pos_box[keepgt1]
                cls_kept_score_gt1 = final_score_gt1[keepgt1]
                for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                    proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
                
                final_score_gt2 = cls_gt1 * (cls_pos_score)**(1/3)
                keepgt2 = nms(cls_pos_box, final_score_gt2, iou_threshold=0.3)
                #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
                cls_kept_box_gt2 = cls_pos_box[keepgt2]
                cls_kept_score_gt2 = final_score_gt2[keepgt2]
                for box2, score2 in zip(cls_kept_box_gt2, cls_kept_score_gt2):
                    proposals_gt2.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})
                
                
            

        
        assert cls_idx == 80

        exp_to_proposals_gt1[exp_id]= proposals_gt1
        exp_to_proposals_gt2[exp_id]= proposals_gt2
        #exp_to_proposals_gt3[exp_id]= proposals_gt3
  
 
    return (exp_to_proposals_gt1,exp_to_proposals_gt2 )
    
def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    #refdb_path = '/home/wj/code/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    print('about to rank proposals via multiprocessing, good luck ~')
    #results = {}
    results = {}
   
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            #print("139.........................%s"%(split))
            # add dataset_split
            sub_args = (idx, args.gpu_id, args.tid, refdb_path, split, args.m ,dataset_splitby)
            #results0[split],results1[split],results2[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
        pool.close()
        pool.join()
    #proposal_dict = {}
    proposal_dict0 = {}
    proposal_dict1 = {}
    proposal_dict2 = {}
    proposal_dict3 = {}
 
    for split in eval_splits:
        assert results[split].successful()
      
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict0[split] = results[split].get()[0]
        proposal_dict1[split] = results[split].get()[1]
        #proposal_dict2[split] = results[split].get()[2]
       
        
    save_path0 = '/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3sqrt_weight_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path1 = '/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3cube_weight_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path2 = '/data1/wj/ref_nms/cache/my_Sigad_clssqrt0.01_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path3 = '/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-8_9_10-20)_proposals_{}_{}.pkl'.format(args.m, args.dataset)
 
    #print('saving proposals to {}...'.format(save_path)) 
    
    with open(save_path0, 'wb') as f:
        pickle.dump(proposal_dict0, f)
        
    with open(save_path1, 'wb') as f:
        pickle.dump(proposal_dict1, f)
    """      
    with open(save_path2, 'wb') as f:
        pickle.dump(proposal_dict2, f)
      
    with open(save_path3, 'wb') as f:
        pickle.dump(proposal_dict3, f)
    """

    print('all done ~')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split-by', default='unc') 
    parser.add_argument('--tid', type=str, default='1031100427')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
