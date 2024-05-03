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
    #json_pth ="/home/wj/code/ref_nms/output/yb008_att_vanilla_refcoco_1224135344.json"
    json_pth ="/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_0913191409.json"
    #json_pth ="/home_data/wj/ref_nms/output/my_bert_attvanilla_refcocog_1115202529.json"
 
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
    #model_path ="/home/wj/code/ref_nms/output/my_nce_att_vanilla_refcocog_0827224102_b.pth"
    #model_path ="/home/wj/code/ref_nms/output/my_sipqkvcat_att_vanilla_refcoco_unc_0903150212_b.pth"
    #model_path = "/home/wj/code/ref_nms/output/my_sipqk_ad_refcoco_0909181253_b.pth"
    #model_path = "/home_data/wj/ref_nms/output/my_sipqk_ad_refcocog_0912233706_b.pth"

    #model_path = "/home/wj/code/ref_nms/output/my_sipsepqk_ad_refcoco_0912234935_b.pth"
    #model_path = "/home_data/wj/ref_nms/output/my_sipqk_ad_att_vanilla_refcoco+_unc_0913094949_b.pth"
    model_path = "/home_data/wj/ref_nms/output/my_bertall_qkad_sigad_refcoco+_unc_1113160151_b.pth"
    
    predictor.load_state_dict(torch.load(model_path,map_location=device))
    #print("69predictorload")
    predictor.to(device)
    predictor.eval()

    #print("41loadok")
    # Rank proposals
    exp_to_proposals_gt0={}
    exp_to_proposals_cls={}
    exp_to_proposals_gt1={}
    exp_to_proposals_gt2={}
    exp_to_proposals_gt3={}
    exp_to_proposals_gt4={}
    exp_to_proposals_gt5={}
    exp_to_proposals_gt={}
    #print("76exp_to_proposals")
    
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
            gtscore, rank_score, = predictor(pos_feat, packed_sent_feats,bert_feats,lfeats)  # [1, n_box]
            #print("56rankscoreshape%s"%(rank_score.shape,))  # [1,n_posbox]
        # Normalize rank score
        #print("53rankscore%s"%rank_score)
        #print("54rankscore[0]%s"%rank_score[0])
        rank_score = torch.sigmoid(rank_score[0])#[n_box]
        gtscore = torch.sigmoid(gtscore[0])        
        #print("54rank_score%s"%rank_score)
        #print("62rank_score shape%s"%(rank_score.shape,))  #torch.Size([304])
        # Split scores and boxes category-wise
        rank_score_list = torch.split(rank_score, cls_num_list, dim=0)  # tuple
        gt_score_list = torch.split(gtscore, cls_num_list, dim=0)  # tuple
        #print("65lenrank_scorelist %s"%(len(rank_score_list)))  #80
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)    # len()=80
        #print("68pos_score_list %s"%(len(pos_score_list)))
        #print("68rank_score shape%s"%(rank_score_list.shape[0],))
        #print("70 pos_scoreshpe%s"%(pos_score.shape,))#[163]
        #print("71class_num_list%s"%cls_num_list )
        #print("72pos_score%s"%(pos_score))
        #print("73pos_score_list%s"%(pos_score_list,))
       
        # Combine score and do NMS category-wise
        proposals_gt0=[]
        proposals_gt1=[]
        proposals_cls=[]
        proposals_gt2=[]
        proposals_gt3=[]
        proposals_gt4=[]
        proposals_gt5=[]
        proposals_gt=[]
        cls_idx = 0
        


        cls_idx = 0
        for cls_gt, cls_ref, cls_pos_box, cls_pos_score in zip(gt_score_list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            #if cls_gt.size(0) == 0  and cls_ref.size(0) == 0: # size(0)some is 0, and some is 16 or other
            if cls_gt.size(0) == 0:
                continue
            # gt cls
            #final_score_gt = cls_gt 
            
                        
            # gt and then cls,
            zero = torch.zeros_like(cls_gt)
            one = 1-zero
            gt_score_001 = torch.where(cls_gt < 0.01, zero, cls_gt)
            cls_pos_score02 = torch.where(cls_pos_score < 0.2,zero,cls_pos_score)
          
            #gt_score_05 = torch.where(gt_score < 0.5, zero, gt_score)
            #gt_score_04 = torch.where(gt_score < 0.4, zero, gt_score)
            final_score_gt = gt_score_001 * cls_pos_score02           
            keepgt = nms(cls_pos_box, final_score_gt, iou_threshold=0.3)
            cls_kept_box_gt0 = cls_pos_box[keepgt]
            cls_kept_score_gt0 = final_score_gt[keepgt]
   
            for box0, score0 in zip(cls_kept_box_gt0, cls_kept_score_gt0):
                proposals_gt0.append({'score': score0.item(), 'box': box0.tolist(), 'cls_idx': cls_idx})
                
            
            cls_pos_score03 = torch.where(cls_pos_score < 0.3,zero,cls_pos_score)
          
            #gt_score_05 = torch.where(gt_score < 0.5, zero, gt_score)
            #gt_score_04 = torch.where(gt_score < 0.4, zero, gt_score)
            final_score_gt1 = gt_score_001 * cls_pos_score03           
            keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
            cls_kept_box_gt1 = cls_pos_box[keepgt1]
            cls_kept_score_gt1 = final_score_gt[keepgt1]
   
            for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            
            """    
            gt_score_001 = torch.where(cls_gt < 0.001, zero, cls_gt) 
            cls_pos_score03 = torch.where(cls_pos_score >= 0.3, one, zero)   
            final_score_gt1 = gt_score_001 * cls_pos_score03
            keepgt1 = nms(cls_pos_box, final_score_gt1, iou_threshold=0.3)
            cls_kept_box_gt1 = cls_pos_box[keepgt1]
            cls_kept_score_gt1 = final_score_gt1[keepgt1]
   
            for box1, score1 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            """
                
            """
            ymax=1
            ymin= 0.3
            cls_gt = (ymax-ymin)*(cls_gt-cls_gt.min())/(cls_gt.max()-cls_gt.min()) + ymin
            #print("159 cls_gt %s"%len(cls_gt))
            #print("160 cls_pos_score %s"%len(cls_pos_score))
            #print("161 final_score_gt...%s"%len(final_score_gt))
            print("176.....cls_gt%s"%cls_gt) 
            
            final_score_gt = cls_gt * cls_pos_score
            keepgt = nms(cls_pos_box, final_score_gt, iou_threshold=0.3)
            cls_kept_box_gt = cls_pos_box[keepgt]
            cls_kept_score_gt = final_score_gt[keepgt]
   
            for box1, score1 in zip(cls_kept_box_gt, cls_kept_score_gt):
                proposals_gt.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            """   
            """
            # only gt
            keepgt = nms(cls_pos_box, final_score_gt, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt = cls_pos_box[keepgt]
            cls_kept_score_gt = final_score_gt[keepgt]                    
            for box0, score0 in zip(cls_kept_box_gt, cls_kept_score_gt):
                proposals_gt0.append({'score': score0.item(), 'box': box0.tolist(), 'cls_idx': cls_idx})
                
                
            # only cls    
            keepcls = nms(cls_pos_box, cls_pos_score, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_cls = cls_pos_box[keepcls]
            cls_kept_score_cls = cls_pos_score[keepcls]                    
            for box1, score1 in zip(cls_kept_box_cls, cls_kept_score_cls):
                proposals_cls.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
                
            #  cls + 2gt    
            gt_ad_cls = 2*cls_gt + cls_pos_score
            #print("167  gt_ad_cls ...%s"% gt_ad_cls )                
            keepgt1 = nms(cls_pos_box, gt_ad_cls, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt1 = cls_pos_box[keepgt1]
            cls_kept_score_gt1 = gt_ad_cls[keepgt1]
            for box2, score2 in zip(cls_kept_box_gt1, cls_kept_score_gt1):
                proposals_gt2.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})
            
            #  2 cls + gt      
            gt_ad_2cls = cls_gt + 2*cls_pos_score
            #print("167  gt_ad_cls ...%s"% gt_ad_cls )                
            keepgt2 = nms(cls_pos_box, gt_ad_2cls, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt2 = cls_pos_box[keepgt2]
            cls_kept_score_gt2 = gt_ad_2cls[keepgt2]
            for box3, score3 in zip(cls_kept_box_gt2, cls_kept_score_gt2):
                proposals_gt3.append({'score': score3.item(), 'box': box3.tolist(), 'cls_idx': cls_idx})
                
            gt_mul_cls = cls_gt*(cls_pos_score).sqrt()
            #print("167  gt_ad_cls ...%s"% gt_ad_cls )                
            keepgt_mul = nms(cls_pos_box, gt_mul_cls, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt_mul = cls_pos_box[keepgt_mul]
            cls_kept_score_gt_mul = gt_mul_cls[keepgt_mul]
            for box4, score4 in zip(cls_kept_box_gt_mul,cls_kept_score_gt_mul):
                proposals_gt4.append({'score': score4.item(), 'box': box4.tolist(), 'cls_idx': cls_idx})
            
           """             
            """
            final_score_ref = cls_ref * cls_pos_score
            keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_ref = cls_pos_box[keepref]
            cls_kept_score_ref = final_score_ref[keepref]

            
            for box1, score1 in zip(cls_kept_box_ref, cls_kept_score_ref):
                proposals_ref.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            

            gt_mul_cls = cls_gt.sqrt() * (cls_pos_score)
            #print("167  gt_ad_cls ...%s"% gt_ad_cls )                
            keepgt_mul = nms(cls_pos_box, gt_mul_cls, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt_mul = cls_pos_box[keepgt_mul]
            cls_kept_score_gt_mul = gt_mul_cls[keepgt_mul]
            for box4, score4 in zip(cls_kept_box_gt_mul,cls_kept_score_gt_mul):
                proposals_gt4.append({'score': score4.item(), 'box': box4.tolist(), 'cls_idx': cls_idx})
                
            gt_mul_cls1 = (cls_pos_score)*(cls_gt**(1/3))
            #print("167  gt_ad_cls ...%s"% gt_ad_cls )                
            keepgt_mul1 = nms(cls_pos_box, gt_mul_cls1, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_gt_mul1 = cls_pos_box[keepgt_mul1]
            cls_kept_score_gt_mul1 = gt_mul_cls1[keepgt_mul1]
            for box5, score5 in zip(cls_kept_box_gt_mul1,cls_kept_score_gt_mul1):
                proposals_gt5.append({'score': score5.item(), 'box': box5.tolist(), 'cls_idx': cls_idx})
        
            """
        assert cls_idx == 80
        
        exp_to_proposals_gt0[exp_id] = proposals_gt0  
        exp_to_proposals_gt1[exp_id] = proposals_gt1
        #exp_to_proposals_gt5[exp_id] = proposals_gt5
        
        """ 
        exp_to_proposals_cls[exp_id] = proposals_cls
        exp_to_proposals_gt2[exp_id] = proposals_gt2
        exp_to_proposals_gt3[exp_id] = proposals_gt3
        exp_to_proposals_gt4[exp_id] = proposals_gt4
        """ 
        
    return (exp_to_proposals_gt0,exp_to_proposals_gt1)

def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    #refdb_path = '/home/wj/code/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    print('about to rank proposals via multiprocessing, good luck ~')
    results = {}
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            print("139%s%s"%(idx,split))
            # add dataset_split
            sub_args = (idx, args.gpu_id, args.tid, refdb_path, split, args.m ,dataset_splitby)
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
        pool.close()
        pool.join()
    
    proposal_dict0 = {}
    proposal_dict1 = {}
    proposal_dict2 = {}
    proposal_dict3 = {}
    proposal_dict4 = {}
    for split in eval_splits:
        assert results[split].successful()
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict0[split] = results[split].get()[0]
        proposal_dict1[split] = results[split].get()[1] 
        """       
        proposal_dict2[split] = results[split].get()[2]
        proposal_dict3[split] = results[split].get()[3]  
        proposal_dict4[split] = results[split].get()[4]
        """
    #save_path = '/home/wj/code/ref_nms/cache/proposalsnocls_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #save_path = '/home/wj/code/ref_nms/cache/ref_ad_08yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path0 = '/home/wj/code/ref_nms/cache/0my_qkad_gtonly_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/home/wj/code/ref_nms/cache/0my_qkad_cls_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path2 = '/home/wj/code/ref_nms/cache/0my_qkad_2gtcls_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path3 = '/home/wj/code/ref_nms/cache/0my_qkad_gt2cls_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path4 = '/home/wj/code/ref_nms/cache/0my_qkad_gtmulcls_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    
    save_path0 = '/data1/wj/ref_nms/cache/my_qkad_gt001mulcls02_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path1 = '/data1/wj/ref_nms/cache/my_qkad_gt001mulcls03_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/home/wj/code/ref_nms/cache/my_qkad_gt0001cls03_score_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    
    #save_path1 = '/home/wj/code/ref_nms/cache/my__ref_proposals_{}_{}.pkl'.format(args.m, args.dataset)
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
    with open(save_path4, 'wb') as f:
        pickle.dump(proposal_dict4, f)    
    """
 
    print('all done ~')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1) 
    parser.add_argument('--dataset', default='refcocog')
    parser.add_argument('--split-by', default='umd')
    parser.add_argument('--tid', type=str, default='0912234935')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
