import json
import pickle
import argparse
from multiprocessing import Pool
import os
import sys
import numpy as np
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from lib.refer import REFER
import torch
import cv2
from tqdm import tqdm
from torchvision.ops import nms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from lib.predictor import AttVanillaPredictorV2
from lib.load_data_clip import DetEvalLoader
from utils.constants import EVAL_SPLITS_DICT
POS_OF_INTEREST = {'NOUN', 'NUM', 'PRON', 'PROPN'}
#from CLIP_main import clip
import clip
# load rpn, >0.05 class confidence; then caculate the relate score, combine the class confidence, obtain the threhold
def rank_proposals(position, gpu_id, tid, refer,refdb_path, split, m):
    # Load refdb
    IMAGE_DIR = '/home/wj/code/MCN_master2/data/images/train2014/'
    with open(refdb_path) as f:
        refdb = json.load(f)
    dataset_ = refdb['dataset_splitby'].split('_')[0]
    # Load pre-trained model
    # device = torch.device('cuda', gpu_id)
    #device = "cuda"
    device = "cpu"
  
    # Load GloVe feature
    with open('/home_data/wj/ref_nms/output/{}_{}_{}.json'.format(m, dataset_, tid), 'r') as f:
        #print("jsonpath%s"%f)
        model_info = json.load(f)
    #predictor = AttVanillaPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
     #                                 rank_dropout_p=model_info['config']['RANK_DROPOUT_P'])
    
    model_path = '/home_data/wj/ref_nms/output/{}_{}_{}_b.pth'.format(m, dataset_, tid)
    #print("model_path%s"%model_path)
    jit_model, transform = clip.load('ViT-B/16', device=device,jit=True)
    #py_model, _ = clip.load('ViT-B/16', device=device,jit=False)
  
    """image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)"""
    """predictor.load_state_dict(torch.load(model_path))
    predictor.to(device)
    predictor.eval()"""
    # Rank proposals
    exp_to_proposals = {}
    loader = DetEvalLoader(refdb, split, gpu_id)
    ref_ids = refer.getRefIds(split=split)
    tqdm_loader = tqdm(loader, desc='scoring {}'.format(split), ascii=True, position=position)
 
    for image_id, exp_id, exps, pos_box,  pos_score,cls_num_list, pos_feat,  pos_box, pos_score in tqdm_loader: # img id,exp id,pos_id,pos score,cls num list
 
 
        # Compute rank score
        #packed_sent_feats = pack_padded_sequence(sent_feat, torch.tensor([sent_feat.size(1)]),
                                                 #enforce_sorted=False, batch_first=True)
                                                   
        #with torch.no_grad():
         # rank_score, *_ = predictor(pos_feat, packed_sent_feats)  # [1, *]
        
        #print("69rankscore %s"%rank_score)
        #print("70rankscoreshape %s"%rank_score.shape)
        #rank_score = torch.sigmoid(rank_score[0])
        #print("72rankscore %s"%rank_score.shape)
        #rank_score_list = torch.split(rank_score, cls_num_list, dim=0)
        #print("74rank_score_list%s"%len(rank_score_list))
        #print("75rank_score_list%s"%rank_score_list)
        #print("56exps%s"%exp_id)
        #sentence = refer.Sents[exp_id]
        #print("58sentence%s"%sentence)
        tokens= refer.sentToTokens[exp_id]
        #print("60tokens%s"%tokens)
        #pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
        #print("77pos_box_list%s"%len(pos_box_list))
        #print("78pos_box_list%s"%(pos_box_list,))
        #pos_score_list = torch.split(pos_score, cls_num_list, dim=0)
        #print("80pos_box_list%s"%len(pos_score_list))
        #print("81pos_box_list%s"%(pos_score_list,))
        img_pth=os.path.join(IMAGE_DIR,'COCO_train2014_000000'+str(image_id)+'.jpg')
        im = cv2.imread(img_pth)
        #im = Image.open(img_pth)
        #im = np.array(im)
        #print("52sent%s box%s"%(packed_sent_feats.shape,pos_feat.shape))
        print("90 whole image%s"%(im.shape,))
        for token in tokens:
          im_box_list=[]
          for n in range(pos_box.shape[0]):
          #print("55n%s"%n)
        #print("53box cor%s"%pos_box[0,:]) # x1 y1 x2 y2
            x1,y1,x2,y2=int(pos_box[n,0]),int(pos_box[n,1]), int(pos_box[n,2])+1,int(pos_box[n,3])+1
        #print("61imshape%s"%(im.shape,)) # (hwc)
            #print("96 x1%s y1%s x2%s y2%s"%(x1,y1,x2,y2))
            im_box = im[y1:y2,x1:x2]
          #im_box = im[x1:x2,y1:y2]
            #im_box_list=im_box_list.append(Image.fromarray(im_box))
            im_box=Image.fromarray(im_box)
            roi_input = transform(im_box).unsqueeze(0).to(device)
            #print("108roishape%s"%(roi_input.shape,))
            #print("109roiinput%s"%(roi_input,))
            re = clip.tokenize(token).to(device)
            with torch.no_grad():
              logits_per_image, _ = jit_model(roi_input, re)
              jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
              #print("119logits_per_image%s"%logits_per_image)
              print("120jit_probs%s"%jit_probs)
              
        # Normalize rank score
        # Split scores and boxes category-wise

        # Combine score and do NMS category-wise
            print("137jit_probs%s"%type(jit_probs))
        #print("55rank_score shape%s"%rank_score.shape)
        # Split scores and boxes category-wise
            rank_score_list = torch.split(jit_probs, cls_num_list, dim=0)  # tuple
            print("58rank_scorelist shape%s"%type(rank_score_list))
            pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
            pos_score_list = torch.split(pos_score, cls_num_list, dim=0)  
        # Combine score and do NMS category-wise
            proposals = []
            cls_idx = 0
          for cls_rank_score, cls_pos_box, cls_pos_score in zip(rank_score_list, pos_box_list, pos_score_list):
            cls_idx += 1
            # No positive box under this category
            if cls_rank_score.size(0) == 0:
                continue
            final_score = cls_rank_score * cls_pos_score
            keep = nms(cls_pos_box, final_score, iou_threshold=0.3)  # according to the final score to rank box, and caculate the iou to filter box
            cls_kept_box = cls_pos_box[keep]
            cls_kept_score = final_score[keep]
            for box, score in zip(cls_kept_box, cls_kept_score):
                proposals.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
          assert cls_idx == 80
          exp_to_proposals[exp_id] = proposals
    return exp_to_proposals
    
def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):

    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    refer = REFER('/home_data/wj/ref_nms/data/refer', args.dataset, args.split_by)
    print('about to rank proposals via multiprocessing, good luck ~')
    results = {}
    device = "cpu"
    #jit_model, transform = clip.load('ViT-B/16', device=device,jit=True)
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            sub_args = (idx, args.gpu_id, args.tid, refer, refdb_path, split, args.m)
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
            #print("95result%s"%results[split])
        pool.close()
        pool.join()
    proposal_dict = {}
    for split in eval_splits:
        assert results[split].successful()
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict[split] = results[split].get()
    save_path = '/data1/wj/ref_nms/cache/clipproposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    print('saving proposals to {}...'.format(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(proposal_dict, f)
    print('all done ~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=3)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--tid', type=str, default='1019204514')
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
