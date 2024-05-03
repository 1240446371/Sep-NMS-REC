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
from lib.my_vanilla_utils1 import DetEvalLoader
#from lib.mypredictor import AttVanillaPredictorV2
#from lib.my_nce_predictor import AttVanillaPredictorV2
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
    json_pth ="/home_data/wj/ref_nms/output/my_bert_attvanilla_refcocog_1115202529.json"
    #json_pth ="/home_data/wj/ref_nms/output/refnce_t1006_refcoco_unc.json"
    #json_pth ="/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_0913191409.json"
    
    
 
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
    #model_path  = "/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_unc_0913191409_b.pth"
    #model_path  = "/home_data/wj/ref_nms/output/my_sipqk_ad_att_vanilla_refcoco+_unc_0913094949_b.pth"
    model_path  = "/home_data/wj/ref_nms/output/my_sipqk_ad_refcocog_0912233706_b.pth"





    


    
    predictor.load_state_dict(torch.load(model_path,map_location=device))
    #print("69predictorload")
    predictor.to(device)
    predictor.eval()

    #print("41loadok")
    # Rank proposals
    exp_to_proposals = {}
    exp_to_proposals_gt={}
    exp_to_proposals_gt1={}
    exp_to_proposals_cls={}
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
        proposals = []
        cls_idx = 0
        proposals_gt=[]
        proposals_gt1=[]
        proposals_cls=[]
        proposals_ref=[]
        cls_idx = 0
        
        for cls_gt, cls_ref, cls_pos_box, cls_pos_score in zip(gt_score_list,rank_score_list,pos_box_list, pos_score_list):
            cls_idx += 1
            #if cls_gt.size(0) == 0  and cls_ref.size(0) == 0: # size(0)some is 0, and some is 16 or other
            if cls_gt.size(0) == 0:
                continue
            # gt cls
                        # gt and then cls,
            zero=torch.zeros_like(cls_gt)
            tag = zero+0.3
            tag1= zero+0.4
            cls_pos_score0 =cls_pos_score
            cls_pos_score1= cls_pos_score
      
            #a=torch.tensor(0.3,device='cuda')
            #b=torch.tensor(0.4,device='cuda')
            #a=torch.tensor(0.3).to(device)
            #b=torch.tensor(0.4).to(device)
            
            gt_score_001 = torch.where(cls_gt < 0.01, zero, cls_gt)
            #cls_pos_score02 = torch.where(cls_pos_score < 0.2,zero,cls_pos_score)
            
            keepgt = nms(cls_pos_box, gt_score_001, iou_threshold=0.3)
            
            tag = tag[keepgt]
          
            #print("188tag%s"%tag)
            #print("179keepgtshape%s keepgt%s"%(keepgt.shape,keepgt))
            #print("179cls_gt%s"%(cls_gt.shape,))
            cls_pos_score0.scatter_add_(0,keepgt,tag)

            #print("zero%s"%zero)

            keepcls = nms(cls_pos_box, cls_pos_score0, iou_threshold=0.3)
            cls_kept_box_cls = cls_pos_box[keepcls]
            cls_kept_score_cls = cls_pos_score0[keepcls]
       
            for box0, score0 in zip(cls_kept_box_cls,cls_kept_score_cls):
                proposals_gt.append({'score': score0.item(), 'box': box0.tolist(), 'cls_idx': cls_idx})
            
            tag1= tag1[keepgt]
            cls_pos_score1.scatter_add_(0,keepgt,tag1)
   
            keepcls1 = nms(cls_pos_box, cls_pos_score1, iou_threshold=0.3)
            cls_kept_box_cls1 = cls_pos_box[keepcls1]
            cls_kept_score_cls1 = cls_pos_score1[keepcls1]
            
       
            for box1, score1 in zip(cls_kept_box_cls1,cls_kept_score_cls1):
                proposals_gt1.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})  
                
            """                
            keepcls = nms(cls_pos_box, cls_pos_score, iou_threshold=0.3)
            #keepref = nms(cls_pos_box, final_score_ref, iou_threshold=0.3)
            cls_kept_box_cls = cls_pos_box[keepcls]
            cls_kept_score_cls = cls_pos_score[keepcls]
            
            for box1, score1 in zip(cls_kept_box_cls,cls_kept_score_cls):
                proposals_cls.append({'score': score1.item(), 'box': box1.tolist(), 'cls_idx': cls_idx})
            """
            
        assert cls_idx == 80
        exp_to_proposals_gt[exp_id] = proposals_gt
        exp_to_proposals_gt1[exp_id] = proposals_gt1
    return (exp_to_proposals_gt,exp_to_proposals_gt1)


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

    for split in eval_splits:
        assert results[split].successful()
       
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict0[split] = results[split].get()[0]
        proposal_dict1[split] = results[split].get()[1]

    #save_path = '/home/wj/code/ref_nms/cache/proposalsnocls_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #save_path = '/home/wj/code/ref_nms/cache/ref_ad_08yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path0 = '/data1/wj/ref_nms/cache/my_qkad_gtnms03_adcls_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path1 = '/data1/wj/ref_nms/cache/my_qkad_gtnms04_adcls_proposals_{}_{}.pkl'.format(args.m, args.dataset)

    
    #print('saving proposals to {}...'.format(save_path)) 
    with open(save_path0, 'wb') as f:
        pickle.dump(proposal_dict0, f)
    with open(save_path1, 'wb') as f:
        pickle.dump(proposal_dict1, f)
  
    print('all done ~')
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', default='refcocog')
    parser.add_argument('--split-by', default='umd')
    parser.add_argument('--tid', type=str, default='0827214414')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
