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


import torch 
from tqdm import tqdm
from torchvision.ops import nms
from torch.nn.utils.rnn import pack_padded_sequence
#import clip
#from lib.predictor import AttVanillaPredictorV2
#from lib.predictort1_ls_3cls import AttVanillaPredictorV2
#from lib.vanilla_utils1 import DetEvalLoader
from lib.my_vanilla_utils import DetEvalLoader
#from lib.mypredictor import AttVanillaPredictorV2
#from lib.my_nce_predictor import AttVanillaPredictorV2
#from lib.my_qkvad_predictor import AttVanillaPredictorV2
#from lib.my_sip_predictor import AttVanillaPredictorV2
from lib.my_qkad_predictor import AttVanillaPredictorV2
from utils.constants import EVAL_SPLITS_DICT

from transformers import AutoTokenizer, BertModel

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
    json_pth ="/home_data/wj/ref_nms/output/refnce_t1006_refcoco_unc.json"
    
    
 
    #with open(json_pth.format(m, dataset_, tid), 'r') as f:
    with open(json_pth,'r') as f:
        #print("jsonpath%s"%f)
        model_info = json.load(f) 
        #print("45jsonpath%s"%f)
    predictor = AttVanillaPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
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
    #model_path  = "/home_data/wj/ref_nms/output/my_qkv_ad_att_vanilla_refcoco_unc_0903230040_b.pth"
    #model_path  = "/home_data/wj/ref_nms/output/my_sipcat_att_vanilla_refcoco_unc_0909180357_b.pth"
    model_path  = "/home_data/wj/ref_nms/output/my_sipqk_ad_att_vanilla_refcoco+_unc_0913094949_b.pth"




    


    
    predictor.load_state_dict(torch.load(model_path,map_location=device))
    #print("69predictorload")
    predictor.to(device)
    predictor.eval()

    #exp_to_proposals = {}
    exp_to_proposals_gt={}
    exp_to_proposals_ref={}
    exp_to_proposals_adgt05={}
    exp_to_proposals_adgt04={}
    exp_to_proposals_adgt049={}
    exp_to_proposals_adgt04nms={}
    #exp_to_proposals_adgtall={}
    #exp_to_proposalsref={}
    exp_to_proposals_adgt05nms={}
    exp_to_proposals_adgt049nms ={}
    #exp_to_proposals_adgt05nms
    
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
    for exp_id,lfeats, pos_feat, sent_feat, bert_feats,pos_box, pos_score, cls_num_list in tqdm_loader:
 
        # Compute rank score
        #t_num+=1
        #print("49t_num%s"%t_num)
        packed_sent_feats = pack_padded_sequence(sent_feat, torch.tensor([sent_feat.size(1)]),
                                                 enforce_sorted=False, batch_first=True)  # save first sequence, then second sequence
        #print("112 pos_feat %s,bert_feats %s,lfeats %s"%(pos_feat.shape,bert_feats.shape,lfeats.shape,))
        with torch.no_grad():
            #print("54pos_featshape%s packed_sent_feats.shape%s"%(pos_feat.shape,packed_sent_feats.shape))
            #rank_score, *_ = predictor(pos_feat, packed_sent_feats)  # [1, n_box]
            # add lfeats
            gt_score , rank_score, = predictor(pos_feat, packed_sent_feats,bert_feats,lfeats)  # [1, n_box]
            #print("56rankscoreshape%s"%(rank_score.shape,))  # [1,n_posbox]
        # Normalize rank score
        #print("53rankscore%s"%rank_score)
        #print("54rankscore[0]%s"%rank_score[0])
        rank_score = torch.sigmoid(rank_score[0])#[n_box]
        gt_score =torch.sigmoid(gt_score[0])
        #zero = torch.zeros_like(gt_score)
        #gt_score_049 = torch.where(gt_score < 0.49, zero, gt_score)
        #gt_score_05 = torch.where(gt_score < 0.5, zero, gt_score)
        #gt_score_04 = torch.where(gt_score < 0.4, zero, gt_score)
        print("54gt_score%s"%gt_score)
        #print("62rank_score shape%s"%(rank_score.shape,))  #torch.Size([304])
        # Split scores and boxes category-wise
        
        #rank_score_gt05 = rank_score + gt_score_05  # add rank and gt > 0.5
        #rank_score_gt04 = rank_score + gt_score_04 
        #rank_score_gt049 = rank_score + gt_score_049 
        
        #rank_score_gtall = rank_score + gt_score
        #gt_score_049list = torch.split(gt_score_049, cls_num_list, dim=0)
        #gt_score_04list = torch.split(gt_score_04, cls_num_list, dim=0)
        #gt_score_05list = torch.split(gt_score_05, cls_num_list, dim=0)
        gt_score_list = torch.split(gt_score, cls_num_list, dim=0)
        #rank_score_gt05_list = torch.split(rank_score_gt05, cls_num_list, dim=0)  # tuple split all box for one cat--list[0] is cat0
        #rank_score_gt04_list = torch.split(rank_score_gt04, cls_num_list, dim=0)
        #rank_score_gt049_list = torch.split(rank_score_gt049, cls_num_list, dim=0)
        #rank_score_gtall_list = torch.split(rank_score_gtall, cls_num_list, dim=0)
        rank_score_list = torch.split(rank_score, cls_num_list, dim=0)
        #80
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)    # len()=80
       
        # Combine score and do NMS category-wise
        """
        proposalsref = []
        cls_idx = 0
        for cls_rank_score, cls_pos_box, cls_pos_score in zip(rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            if cls_rank_score.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            final_score = cls_rank_score * cls_pos_score
            keep = nms(cls_pos_box, final_score, iou_threshold=0.3)
            cls_kept_box = cls_pos_box[keep]
            cls_kept_score = final_score[keep]       
            for box, score in zip(cls_kept_box, cls_kept_score):
                proposalsref.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposalsref[exp_id] = proposalsref
        
        proposalsgt = []
        cls_idx = 0
        for cls_gt_score, cls_pos_box, cls_pos_score in zip(gt_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            if cls_gt_score.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            final_gt_score = cls_gt_score * cls_pos_score
            keepgt = nms(cls_pos_box, final_gt_score, iou_threshold=0.3)
            cls_kept_boxgt = cls_pos_box[keepgt]
            cls_kept_scoregt = final_gt_score[keepgt]       
            for box, score in zip(cls_kept_boxgt, cls_kept_scoregt):
                proposalsgt.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposalsgt[exp_id] = proposalsgt
        """      
        proposals_adgt04 = []
        proposals_adgt049 = []
        proposals_adgt05 = []
        proposals_adgtnms04 = []
        proposals_adgtnms049 = []
        proposals_adgtnms05 = []
        proposals_gt=[]
        proposals_ref=[]
        cls_idx = 0
        for cls_gt, cls_ref, cls_pos_box, cls_pos_score in zip(gt_score_list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            #if cls_gt.size(0) == 0  and cls_ref.size(0) == 0: # size(0)some is 0, and some is 16 or other
            if cls_gt.size(0) == 0:
                continue
            # gt cls
            final_score_gt = cls_gt * cls_pos_score
            #final_score_ref = cls_ref * cls_pos_score
            #zero = torch.zeros_like(final_score_gt)
            #gt_score_049 = torch.where(gt_score < 0.49, zero, gt_score)
            #final_score_gt04=torch.where(final_score_gt < 0.4, zero, final_score_gt)
            #final_score_gt049=torch.where(final_score_gt < 0.49, zero, final_score_gt)
            #final_score_gt05=torch.where(final_score_gt < 0.5, zero, final_score_gt)
            
            # (ref + gt*cls) * cls then nms
            #cls_ref_adgt04 =(cls_ref+final_score_gt04)* cls_pos_score
            #cls_ref_adgt049 =(cls_ref+final_score_gt049)* cls_pos_score
            #cls_ref_adgt05 =(cls_ref+final_score_gt05)*cls_pos_score
            # ref ad gt nms
            #keepad04 = nms(cls_pos_box, cls_ref_adgt04, iou_threshold=0.3)
            #keepad049 = nms(cls_pos_box, cls_ref_adgt049, iou_threshold=0.3)
            #keepad05 = nms(cls_pos_box, cls_ref_adgt05, iou_threshold=0.3)


            #cls_kept_box_adgt04 = cls_pos_box[keepad04]
            #cls_kept_score_adgt04 = cls_ref_adgt04[keepad04]              
    
            #cls_kept_box_adgt049 = cls_pos_box[keepad049]
            #cls_kept_score_adgt049 = cls_ref_adgt049[keepad049]   
                   
            #cls_kept_box_adgt05 = cls_pos_box[keepad05]
            #cls_kept_score_adgt05 = cls_ref_adgt05[keepad05]   
                 
            # gt nms
            #keepgt04 = nms(cls_pos_box, final_score_gt04, iou_threshold=0.3)
            #keepgt049 = nms(cls_pos_box, final_score_gt049, iou_threshold=0.3)
            #keepgt05 = nms(cls_pos_box, final_score_gt05, iou_threshold=0.3)
            
            # (ref + gtnms)*cls  then  nms 
            #cls_ref0= cls_ref
            #cls_ref1= cls_ref
            #cls_ref2 = cls_ref
            
            #cls_kept_score_gtnms04 = final_score_gt04[keepgt04]
            #cls_ref_ad_gtnms04 =  cls_ref0.scatter_add_(0, keepgt04, cls_kept_score_gtnms04)
            #final_cls_ref_ad_gtnms04 = cls_ref_ad_gtnms04 * cls_pos_score
            #keep_ad_nms04 = nms(cls_pos_box, final_cls_ref_ad_gtnms04, iou_threshold=0.3)
            #cls_kept_box_nms04 = cls_pos_box[keep_ad_nms04]
            #cls_kept_score_nms04 = final_cls_ref_ad_gtnms04[keep_ad_nms04]     
            """
            cls_kept_score_gtnms049 = final_score_gt049[keepgt049] 
            cls_ref_ad_gtnms049 =  cls_ref1.scatter_add_(0, keepgt049, cls_kept_score_gtnms049)
            final_cls_ref_ad_gtnms049 = cls_ref_ad_gtnms049 * cls_pos_score
            keep_ad_nms049 = nms(cls_pos_box, final_cls_ref_ad_gtnms049, iou_threshold=0.3)
            cls_kept_box_nms049 = cls_pos_box[keep_ad_nms049]
            cls_kept_score_nms049 = final_cls_ref_ad_gtnms049[keep_ad_nms049] 
               
            cls_kept_score_gtnms05 = final_score_gt05[keepgt05] 
            cls_ref_ad_gtnms05 =  cls_ref2.scatter_add_(0, keepgt05, cls_kept_score_gtnms05) 
            final_cls_ref_ad_gtnms05 = cls_ref_ad_gtnms05 * cls_pos_score
            keep_ad_nms05= nms(cls_pos_box, final_cls_ref_ad_gtnms05, iou_threshold=0.3)
            cls_kept_box_nms05 = cls_pos_box[keep_ad_nms05]
            cls_kept_score_nms05 = final_cls_ref_ad_gtnms05[keep_ad_nms05]     

            for box0, score0 in zip(cls_kept_box_adgt04, cls_kept_score_adgt04):
                proposals_adgt04.append({'score': score0.item(), 'box': box0.tolist(), 'cls_idx': cls_idx})
            for box1, score1 in zip(cls_kept_box_adgt049, cls_kept_score_adgt049):
                proposals_adgt049.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            for box2, score2 in zip(cls_kept_box_adgt05, cls_kept_score_adgt05):
                proposals_adgt05.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})
                
            for box3, score3 in zip(cls_kept_box_nms04, cls_kept_score_nms04):
                proposals_adgtnms04.append({'score': score3.item(), 'box': box3.tolist(), 'cls_idx': cls_idx})
            for box4, score4 in zip(cls_kept_box_nms049, cls_kept_score_nms049):
                proposals_adgtnms049.append({'score': score4.item(), 'box': box4.tolist(), 'cls_idx': cls_idx})
            for box5, score5 in zip(cls_kept_box_nms05, cls_kept_score_nms05):
                proposals_adgtnms05.append({'score': score5.item(), 'box': box5.tolist(), 'cls_idx': cls_idx})
            """
            keepgt = nms(cls_pos_box, final_score_gt, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt = cls_pos_box[keepgt]
            cls_kept_score_gt = final_score_gt[keepgt]
            #cls_kept_box_ref = cls_pos_box[keepref]
            #cls_kept_score_ref = final_score_ref[keepref]
            
            for box0, score0 in zip(cls_kept_box_gt, cls_kept_score_gt):
                proposals_gt.append({'score': score0.item(), 'box': box0.tolist(), 'cls_idx': cls_idx})
            """
            for box1, score1 in zip(cls_kept_box_ref, cls_kept_score_ref):
                proposals_ref.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            """ 
            
        assert cls_idx == 80
        exp_to_proposals_gt[exp_id] = proposals_gt
        #exp_to_proposals_ref[exp_id]= proposals_ref
        
        """
        exp_to_proposals_adgt04[exp_id] = proposals_adgt04
        exp_to_proposals_adgt049[exp_id]= proposals_adgt049
        exp_to_proposals_adgt05[exp_id] = proposals_adgt05
        exp_to_proposals_adgt04nms[exp_id]= proposals_adgtnms04
        exp_to_proposals_adgt049nms[exp_id] = proposals_adgtnms049
        exp_to_proposals_adgt05nms[exp_id]=proposals_adgtnms05
        # wrong
        proposals_adgt049 = []
        cls_idx = 0
        for cls_rank_gt049, cls_pos_box, cls_pos_score in zip(rank_score_gt049_list,pos_box_list, pos_score_list):
            cls_idx += 1
            if cls_rank_gt049.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            final_score_gt049 = cls_rank_gt049 * cls_pos_score
            keep049 = nms(cls_pos_box, final_score_gt049, iou_threshold=0.3)
            cls_kept_box_adgt049 = cls_pos_box[keep049]
            cls_kept_score_adgt049 = final_score_gt049[keep049]        
            for box, score in zip(cls_kept_box_adgt049, cls_kept_score_adgt049):
                proposals_adgt049.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgt049[exp_id] = proposals_adgt049
        
        proposals_adgt05 = []
        cls_idx = 0
        for cls_rank_gt05, cls_pos_box, cls_pos_score in zip(rank_score_gt05_list,pos_box_list, pos_score_list):
            cls_idx += 1
            if cls_rank_gt05.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            final_score_gt05 = cls_rank_gt05 * cls_pos_score
            keep05 = nms(cls_pos_box, final_score_gt05, iou_threshold=0.3)
            #print("226kepp.....%s"%(keep05.shape,))# 
            #print("227cls_pos_box%s"%(cls_pos_box.shape,))  #(181,4)
            #print("230cls_pos_score05%s"%(cls_pos_score.shape,))  # (41,4)
            cls_kept_box_adgt05 = cls_pos_box[keep05]
            #print("232cls_kept_box_adgt05%s"%(cls_kept_box_adgt05.shape,))  # (41)
            cls_kept_score_adgt05 = final_score_gt05[keep05]        
            for box, score in zip(cls_kept_box_adgt05, cls_kept_score_adgt05):
                proposals_adgt05.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgt05[exp_id] = proposals_adgt05
       
        proposals_adgt04nms = []
        cls_idx = 0
        for cls_rank_gt04, cls_rank_score0, cls_pos_box, cls_pos_score in zip(gt_score_04list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            
            #if cls_rank_gt05.size(0) == 0: # size(0)some is 0, and some is 16 or other
            #    continue
            final_score_gt04 = cls_rank_gt04 * cls_pos_score
            keepgt04 = nms(cls_pos_box, final_score_gt04, iou_threshold=0.3)
            #print("242kepp.....%s"%keep04)
            #cls_kept_box_adgt04nms = cls_pos_box[keep04]
            cls_kept_score_gt04nms = final_score_gt04[keepgt04] 
            print("252....keep04 %s "%keepgt04)
            print("253....cls_rank_score0 %s "%cls_rank_score0)
            print("254... cls_kept_score_gt04nms %s "% cls_kept_score_gt04nms)
            cls_rank_ad_gtnms =  cls_rank_score0.scatter_add_(0, keepgt04, cls_kept_score_gt04nms)
            print("256....cls_rank_ad_gtnms %s "%cls_rank_ad_gtnms)
            final_cls_score_ad_gtnms04 = cls_rank_ad_gtnms * cls_pos_score
            keepnms04 = nms(cls_pos_box, final_cls_score_ad_gtnms04, iou_threshold=0.3)
            cls_kept_box_nms04 = cls_pos_box[keepnms04]
            cls_kept_score_nms04 = final_cls_score_ad_gtnms04[keepnms04]     

            for box, score in zip(cls_kept_box_nms04, cls_kept_score_nms04):
                proposals_adgt04nms.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgt04nms[exp_id] = proposals_adgt04nms
        
        
        
        proposals_adgt049nms = []
        cls_idx = 0
        for cls_rank_gt049, cls_rank_score1, cls_pos_box, cls_pos_score in zip(gt_score_049list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            
            #if cls_rank_gt05.size(0) == 0: # size(0)some is 0, and some is 16 or other
            #    continue
            final_score_gt049 = cls_rank_gt049 * cls_pos_score
            keepgt049 = nms(cls_pos_box, final_score_gt049, iou_threshold=0.3)
            #print("242kepp.....%s"%keep04)
            #cls_kept_box_adgt04nms = cls_pos_box[keep04]
            cls_kept_score_gt049nms = final_score_gt049[keepgt049] 
            #print("252....keep04 %s "%keepgt04)
            #print("253....cls_rank_score %s "%cls_rank_score)
            #print("254... cls_kept_score_gt04nms %s "% cls_kept_score_gt04nms)
            cls_score_ad_gtnms049 =  cls_rank_score1.scatter_add_(0, keepgt049, cls_kept_score_gt049nms)
            #print("256....cls_rank_ad_gtnms %s "%cls_rank_ad_gtnms)
            final_cls_score_ad_gtnms049 = cls_score_ad_gtnms049 * cls_pos_score
            keepnms049 = nms(cls_pos_box, final_cls_score_ad_gtnms049, iou_threshold=0.3)
            cls_kept_box_nms049 = cls_pos_box[keepnms049]
            cls_kept_score_nms049 = final_cls_score_ad_gtnms049[keepnms049]     

            for box, score in zip(cls_kept_box_nms049, cls_kept_score_nms049):
                proposals_adgt049nms.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgt049nms[exp_id] = proposals_adgt049nms
        
        proposals_adgt05nms = []
        cls_idx = 0
        for cls_rank_gt05, cls_rank_score2, cls_pos_box, cls_pos_score in zip(gt_score_05list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            
            #if cls_rank_gt05.size(0) == 0: # size(0)some is 0, and some is 16 or other
            #    continue
            final_score_gt05 = cls_rank_gt05 * cls_pos_score
            keepgt05 = nms(cls_pos_box, final_score_gt05, iou_threshold=0.3)
            #print("242kepp.....%s"%keep04)
            #cls_kept_box_adgt04nms = cls_pos_box[keep04]
            cls_kept_score_gt05nms = final_score_gt05[keepgt05] 
            #print("252....keep04 %s "%keepgt04)
            #print("253....cls_rank_score %s "%cls_rank_score)
            #print("254... cls_kept_score_gt04nms %s "% cls_kept_score_gt04nms)
            cls_score_ad_gtnms05 =  cls_rank_score2.scatter_add_(0, keepgt05, cls_kept_score_gt05nms)
            #print("256....cls_rank_ad_gtnms %s "%cls_rank_ad_gtnms)
            final_cls_score_ad_gtnms05 = cls_score_ad_gtnms05 * cls_pos_score
            keepnms05 = nms(cls_pos_box, final_cls_score_ad_gtnms05, iou_threshold=0.3)
            cls_kept_box_nms05 = cls_pos_box[keepnms05]
            cls_kept_score_nms05 = final_cls_score_ad_gtnms05[keepnms05]     

            for box, score in zip(cls_kept_box_nms05, cls_kept_score_nms05):
                proposals_adgt05nms.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgt05nms[exp_id] = proposals_adgt05nms
        
        
        
        proposals_adgtall = []
        cls_idx = 0
        for cls_rank_gtall, cls_pos_box, cls_pos_score in zip(rank_score_gtall_list,pos_box_list, pos_score_list):
            cls_idx += 1
            if cls_rank_gtall.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            #final_score = cls_rank_score * cls_pos_score  # add cls conf
            final_score_gtall = cls_rank_gtall * cls_pos_score
            #keep = nms(cls_pos_box, final_score, iou_threshold=0.3)  # according to the final score to rank box, and caculate the iou to filter box
            keepgtall = nms(cls_pos_box, final_score_gtall, iou_threshold=0.3)
            #cls_kept_box = cls_pos_box[keep]
            #cls_kept_score = final_score[keep] 
            cls_kept_box_adgtall = cls_pos_box[keepgtall]
            cls_kept_score_adgtall = final_score_gtall[keepgtall] 
            
            for box, score in zip(cls_kept_box_adgtall, cls_kept_score_adgtall):
                proposals_adgtall.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals_adgtall[exp_id] = proposals_adgtall
        #print("196..exp_to_proposalsgt%s,exp_to_proposals_adgt05%s,exp_to_proposals_adgtall%s"%(exp_to_proposalsgt,exp_to_proposals_adgt05,exp_to_proposals_adgtall,))
        """ 
    #return (exp_to_proposalsref,exp_to_proposalsgt,exp_to_proposals_adgt05,exp_to_proposals_adgtall)
    #return (exp_to_proposals_adgt04,exp_to_proposals_adgt049,exp_to_proposals_adgt05,exp_to_proposals_adgt04nms,exp_to_proposals_adgt049nms,exp_to_proposals_adgt05nms)
    #return (exp_to_proposals_adgt04nms,exp_to_proposals_adgt049nms,exp_to_proposals_adgt05nms)
    #return (exp_to_proposals_gt,exp_to_proposals_ref)
    return exp_to_proposals_gt
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
            #print("139%s%s"%(idx,split))
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
    proposal_dict4 = {}
    proposal_dict5 = {}
    for split in eval_splits:
        assert results[split].successful()
       
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict0[split] = results[split].get()
        #proposal_dict1[split] = results[split].get()[1]
        #proposal_dict2[split] = results[split].get()[2]
        #proposal_dict3[split] = results[split].get()[3]
        #proposal_dict4[split] = results[split].get()[4]
        #proposal_dict5[split] = results[split].get()[5]
    #save_path = '/home/wj/code/ref_nms/cache/proposalsnocls_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #save_path = '/home/wj/code/ref_nms/cache/ref_ad_08yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path0 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt04_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/data1/wj/ref_nms/cache/my_refsip_gt_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt049_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path2 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt05_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path3 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt04nms_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path4 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt049nms_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path5 = '/data1/wj/ref_nms/cache/0my_refsipsep_adgt05nms_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path3 = '/data1/wj/ref_nms/cache/my_refsip_adgtall_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path2 = '/data1/wj/ref_nms/cache/my_refsip_adgt02_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path3 = '/data1/wj/ref_nms/cache/my_refsip_adgtall_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path0 = '/data1/wj/ref_nms/cache/my_sipqk_gt_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/data1/wj/ref_nms/cache/my_sipcat_ref_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    
    #print('saving proposals to {}...'.format(save_path)) 
   
    with open(save_path0, 'wb') as f:
        pickle.dump(proposal_dict0, f)
    #with open(save_path1, 'wb') as f:
    #    pickle.dump(proposal_dict1, f)
        
    """
    with open(save_path2, 'wb') as f:
        pickle.dump(proposal_dict2, f)
    with open(save_path3, 'wb') as f:
        pickle.dump(proposal_dict3, f)
    with open(save_path4, 'wb') as f:
        pickle.dump(proposal_dict4, f)
    with open(save_path5, 'wb') as f:
        pickle.dump(proposal_dict5, f)
    """
    print('all done ~')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split-by', default='unc') 
    parser.add_argument('--tid', type=str, default='0827214414')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
