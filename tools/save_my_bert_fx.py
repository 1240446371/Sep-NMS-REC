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
    exp_to_proposals_gt11={}
    exp_to_proposals_gt2={}
    exp_to_proposals_gt3={}
    exp_to_proposals_ref={}  
    exp_to_proposals_adgtall={}
  
    
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
        gt_score_fx1 = (gt_score[0]).cpu().numpy()
        gt_score_fx11 = (gt_score[0]).cpu().numpy()
        #print("133.....")
        gt_score_fx2 = gt_score[0].cpu().numpy()
        #gt_score_fx22 = gt_score[0].cpu().numpy()
        
        gt_score_fx3 = gt_score[0].cpu().numpy()
        #gt_score_fx33 = gt_score[0].cpu().numpy()
        #print("gt_score_fx1device%s"%(gt_score_fx1.device()))
        """
        bins1 = torch.tensor([0.0, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
        labels1=['0', '1', '2','3','4','5']
        
        bins2 = [0.0, 0.01, 0.4, 0.7, 1.0]
        labels2=['0', '1', '2','3']
        
        bins3 = [0.0, 0.01, 0.2, 1.0]      
        labels3=['0', '1', '2'] 
        
        cats1 = pd.cut(gt_score_fx1, bins1,labels1)
        cats2 = pd.cut(gt_score_fx2, bins2,labels2)
        cats3 = pd.cut(gt_score_fx3, bins3,labels3)
        """
        labels1=['0', '1', '2','3','4','5','6','7','8','9']
        cats1 = pd.qcut(gt_score_fx1, 10 ,duplicates='drop',labels= labels1)
        print("159............")

        
        
        j=0
        for i in cats1.codes:
           if i ==0:
             gt_score_fx1[j]=0.1
           if i ==1:
             gt_score_fx1[j]=0.2
           if i ==2:
             gt_score_fx1[j]=0.3
           if i ==3:
             gt_score_fx1[j]=0.4
           if i ==4:
             gt_score_fx1[j]=0.5
           if i ==5:
             gt_score_fx1[j]=0.6             
           if i ==6:
             gt_score_fx1[j]=0.7
           if i ==7:
             gt_score_fx1[j]=0.8
           if i ==8:
             gt_score_fx1[j]=0.9
           if i ==9:
             gt_score_fx1[j]=1
           j+=1
           
        jj=0
        for ii in cats1.codes:
           if ii ==0:
             gt_score_fx11[jj]=0.
           if ii ==1:
             gt_score_fx11[jj]=0.
           if ii ==2:
             gt_score_fx11[jj]=0.
           if ii ==3:
             gt_score_fx11[jj]=0.
           if ii ==4:
             gt_score_fx11[jj]=0.01
           if ii ==5:
             gt_score_fx11[jj]=0.2             
           if ii ==6:
             gt_score_fx11[jj]=0.4
           if ii ==7:
             gt_score_fx11[jj]=0.6
           if ii ==8:
             gt_score_fx11[jj]=0.8
           if ii ==9:
             gt_score_fx11[jj]=1
           jj+=1
           
                  
        labels2=['0', '1', '2','3','4']
        cats2 = pd.qcut(gt_score_fx2, 5 ,duplicates='drop',labels= labels2)
        print("214............")

        m=0
        for a in cats2.codes:
           if a ==0: #0.2
             gt_score_fx2[m]=0.0
           if a ==1:  #0.4
             gt_score_fx2[m]=0.000
           if a ==2:  # 0.6
             gt_score_fx2[m]=0.35
           if a ==3:  #0.8
             gt_score_fx2[m]=0.7
           if a ==4:  # 1
             gt_score_fx2[m]=1
   
           m+=1       
        # 0-0.01-->0.01 , 0.01-0.2-->0.2, else donot change     
         
        
        labels3=['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
        cats3 = pd.qcut(gt_score_fx3, 20 ,duplicates='drop',labels= labels3)
        print("235............")
        n=0 
        for b in cats3.codes:
           #print("b///////%s"%type(b))
           if b == 0 or b ==1 or b ==2 or b ==3 or b ==4 or b ==5 or b ==6 or b ==7  or b ==8 :
             gt_score_fx3[n]=0.0
             
           if b ==9:  # 45-50
             gt_score_fx3[n]=0.01            
           if b ==10: 
             gt_score_fx3[n]=0.1
           if b ==11:
             gt_score_fx3[n]=0.2
           if b ==12:
             gt_score_fx3[n]=0.3
           if b ==13:
             gt_score_fx3[n]=0.4             
           if b ==14:
             gt_score_fx3[n]=0.5             
           if b ==15:
             gt_score_fx3[n]=0.6                      
           if b ==16:
             gt_score_fx3[n]=0.7
           if b ==17:
             gt_score_fx3[n]=0.8
           if b ==18:
             gt_score_fx3[n]=0.9
           if b ==19:
             gt_score_fx3[n]=1

           n+=1             
        print("gt_score_fx11%s"%gt_score_fx11)
        #print("187gt_score_fx3%s"%(gt_score_fx3,))
        gt_score_fx1=torch.from_numpy(gt_score_fx1).to(device)
        gt_score_fx11=torch.from_numpy(gt_score_fx11).to(device)
        gt_score_fx2=torch.from_numpy(gt_score_fx2).to(device)
        gt_score_fx3=torch.from_numpy(gt_score_fx3).to(device)
        gt_fx1_list = torch.split(gt_score_fx1, cls_num_list, dim=0)  # tuple    
        gt_fx11_list = torch.split(gt_score_fx11, cls_num_list, dim=0)  # tuple    
        gt_fx2_list = torch.split(gt_score_fx2, cls_num_list, dim=0)  # tuple   
        gt_fx3_list = torch.split(gt_score_fx3, cls_num_list, dim=0)  # tuple      
        #gt_ad_ref =  gt_score[0] + rank_score[0]
       
        #gt_ad_ref_list = torch.split(gt_ad_ref, cls_num_list, dim=0)  
              
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)        
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)    # len()=80
        #print("138.........")
       
        # Combine score and do NMS category-wise

   
        proposals_gt1=[]
        proposals_gt11=[]
        proposals_gt2=[]
        proposals_gt3=[]
        
        proposals_ref=[]
        proposals_adgtall=[]
      
        cls_idx = 0
        for cls_gt1,cls_gt11, cls_gt2, cls_gt3,cls_pos_box, cls_pos_score in zip(gt_fx1_list, gt_fx11_list, gt_fx1_list,gt_fx3_list, pos_box_list, pos_score_list):
            #print("144ok")
            cls_idx += 1
            if cls_gt1.size(0) == 0 and  cls_gt11.size(0) == 0 and cls_gt2.size(0) == 0 and cls_gt3.size(0) == 0: # size(0)some is 0, and some is 16 or other
                continue
            # gt cls
          
            final_score_gt1 = cls_gt1 * cls_pos_score
            keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt1 = cls_pos_box[keepgt1]
            cls_kept_score_gt1 = final_score_gt1[keepgt1]
            for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            
                      
            final_score_gt11 = cls_gt11 * cls_pos_score
            keepgt11 = nms(cls_pos_box, final_score_gt11, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt11 = cls_pos_box[keepgt11]
            cls_kept_score_gt11 = final_score_gt11[keepgt11]
            for box11, score11 in zip(cls_kept_box_gt11, cls_kept_score_gt11):
                proposals_gt11.append({'score': score11.item(), 'box': box11.tolist(), 'cls_idx': cls_idx})
                
            final_score_gt2 = cls_gt2 * cls_pos_score
            keepgt2 = nms(cls_pos_box, final_score_gt2, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt2 = cls_pos_box[keepgt2]
            cls_kept_score_gt2 = final_score_gt2[keepgt2]
            for box2, score2 in zip(cls_kept_box_gt2, cls_kept_score_gt2):
                proposals_gt2.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})       
                
            final_score_gt3 = cls_gt3 * cls_pos_score
            keepgt3 = nms(cls_pos_box, final_score_gt3, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt3 = cls_pos_box[keepgt3]
            cls_kept_score_gt3 = final_score_gt3[keepgt3]
            for box3, score3 in zip(cls_kept_box_gt3, cls_kept_score_gt3):
                proposals_gt3.append({'score': score3.item(), 'box': box3.tolist(), 'cls_idx': cls_idx})         
     
        
        assert cls_idx == 80

        exp_to_proposals_gt1[exp_id]= proposals_gt1
        exp_to_proposals_gt11[exp_id]= proposals_gt11
        exp_to_proposals_gt2[exp_id]= proposals_gt2
        exp_to_proposals_gt3[exp_id]= proposals_gt3

        
 
    return (exp_to_proposals_gt1,exp_to_proposals_gt11 ,exp_to_proposals_gt2,exp_to_proposals_gt3)
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
 
    for split in eval_splits:
        assert results[split].successful()
      
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict0[split] = results[split].get()[0]
        proposal_dict1[split] = results[split].get()[1]
        proposal_dict2[split] = results[split].get()[2]
        proposal_dict3[split] = results[split].get()[3]
        
    save_path0 = '/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0_9)_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path1 = '/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-3_4_5-9)_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path2 = '/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-1_2-5)_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path3 = '/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-8_9_10-20)_proposals_{}_{}.pkl'.format(args.m, args.dataset)
 
    #print('saving proposals to {}...'.format(save_path)) 
    """
    with open(save_path0, 'wb') as f:
        pickle.dump(proposal_dict0, f)
        
    with open(save_path1, 'wb') as f:
        pickle.dump(proposal_dict1, f)
        
    with open(save_path2, 'wb') as f:
        pickle.dump(proposal_dict2, f)
        
    with open(save_path3, 'wb') as f:
        pickle.dump(proposal_dict3, f)
    """

    print('all done ~')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split-by', default='unc') 
    parser.add_argument('--tid', type=str, default='1031100427')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
