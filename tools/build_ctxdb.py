import json

import numpy as np
import spacy
from tqdm import tqdm
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from lib.refer import REFER
from utils.constants import CAT_ID_TO_NAME, EVAL_SPLITS_DICT
from utils.misc import xywh_to_xyxy, calculate_iou
from io import BytesIO as StringIO

POS_OF_INTEREST = {'NOUN', 'NUM', 'PRON', 'PROPN'}

# txt contain: re , gt box and their cat id ,cxt box and their cat id ,caculate all ann_cat glove and noun_glove sim,and >0.4 is cxt box
# load glove into dict(float)
def load_glove_feats():
    glove_path = '/home_data/wj/ref_nms/data/glove.840B.300d/glove.840B.300d.txt'
    #print('loading GloVe feature from {}'.format(glove_path))
    glove_dict = {}
    with open(glove_path, 'r') as f:
        with tqdm(total=2196017, desc='Loading GloVe', ascii=True) as pbar:  #progressbar
            for line in f:
                tokens = line.split(' ')
                assert len(tokens) == 301  #if len=301,continue, other stop
                word = tokens[0]
                vec = list(map(lambda x: float(x), tokens[1:])) #transfer token into float
                glove_dict[word] = vec # dic, feats
                pbar.update(1)
    return glove_dict


def cosine_similarity(feat_a, feat_b):
    return np.sum(feat_a * feat_b) / np.sqrt(np.sum(feat_a * feat_a) * np.sum(feat_b * feat_b))


def build_ctxdb(dataset, split_by):
    dataset_splitby = '{}_{}'.format(dataset, split_by)
    # Load refer
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset, split_by)
    # Load GloVe feature
    glove_dict = load_glove_feats()
    # Construct COCO category GloVe feature
    cat_id_to_glove = {}
    for cat_id, cat_name in CAT_ID_TO_NAME.items():  #load cat's feats in glove_dict
        cat_id_to_glove[cat_id] = [np.array(glove_dict[t], dtype=np.float32) for t in cat_name.split(' ')]  # obtain box cat glove feature
    # Spacy to extract POS tags
    nlp = spacy.load('en_core_web_sm')
    # Go through the refdb
    ctxdb = {}
    """sentence: [{'tokens': ['fingers', 'on', 'the', 'left', 'touching', 'the', 'sandwich'], 'raw': 'fingers on the left touching the sandwich', 'sent_id': 5884, 'sent': 'fingers on the left touching the sandwich'},
              {'tokens': ['hand', 'sticking', 'in', 'frame', 'on', 'left'], 'raw': 'hand(?) sticking in frame on left', 'sent_id': 5885, 'sent': 'hand sticking in frame on left'}, 
              {'tokens': ['the', 'hand', 'on', 'left', 'cut', 'off'], 'raw': 'the hand on left cut off', 'sent_id': 5886, 'sent': 'the hand on left cut off'}]"""
              
#67sent:(one element of sentence) {'tokens': ['left', 'leaf', 'thing'], 'raw': 'left leaf thing', 'sent_id': 5882, 'sent': 'left leaf thing'}

#sent[sent] (is  sentence in sent) :'left leaf thing' 

#noun tokens is : ['girl', 'right']
    for split in (['train'] + EVAL_SPLITS_DICT[dataset_splitby]):
        exp_to_ctx = {}
        gt_miss_num, empty_num, sent_num = 0, 0, 0
        coco_box_num_list, ctx_box_num_list = [], []
        ref_ids = refer.getRefIds(split=split)
        print("69 split %s"%split)
        #print("69ref_ids %s"%ref_ids)
        for ref_id in tqdm(ref_ids, ascii=True, desc=split):
        
            ref = refer.Refs[ref_id]
            #print("74ref%s"%ref)
            image_id = ref['image_id']
            #print("76imgid%s"%image_id)
            gt_box = xywh_to_xyxy(refer.Anns[ref['ann_id']]['bbox'])
            gt_cat = refer.Anns[ref['ann_id']]['category_id']  
            for sent in ref['sentences']:
                #print("66 sentence%s"% len(ref['sentences']))
                #print("67sent %s"%sent)
                #print("68sent['sent']%s"%sent['sent'])
                
                sent_num += 1
                sent_id = sent['sent_id']
                doc = nlp(sent['sent'])
                noun_tokens = [token.text for token in doc if token.pos_ in POS_OF_INTEREST]
                #print("74tokentext%s"%token.text)
                #print("65tokentext%s"%token.text)
                #print('67SENT', sent['sent']) 
                 # 67SENT image of bottle on left68NOUN TOKENS ['image', 'bottle']
                 # 67SENT left bottle68NOUN TOKENS ['bottle']

                #print('80NOUN TOKENS', noun_tokens)
                noun_glove_list = [np.array(glove_dict[t], dtype=np.float32) for t in noun_tokens if t in glove_dict]  # get noun_token's glove feats
                gt_hit = False
                ctx_list = []
#96ann{'segmentation': [[478.36, 78.27, 449.22, 109.65, 432.42, 179.13, 423.45, 231.79, 430.17, 276.62, 423.45, 291.19, 425.69, 355.06, 436.9, 353.94, #439.14, 367.39, 452.59, 366.27, 452.59, 372.99, 435.78, 384.2, 425.69, 392.04, 416.73, 444.71, 416.73, 506.34, 438.02, 525.39, 451.47, 513.06, 459.31, #463.76, 463.79, 416.69, 470.52, 393.16, 477.24, 387.56, 479.48, 340.49, 480.0, 337.29, 480.0, 291.35, 478.6, 217.39, 477.48, 175.93]], 'area': 18847
#.397299999997, 'iscrowd': 0, 'image_id': 579382, 'bbox': [416.73, 78.27, 63.27, 447.12], 'category_id': 1, 'id': 201995}

                for ann in refer.imgToAnns[image_id]:  #ann has many 7 annotations include cat_id 
                    #print("96ann%s"%ann)
                    #print("100lenann%s"%len(refer.imgToAnns[image_id]))
                    ann_glove_list = cat_id_to_glove[ann['category_id']]  # ann box's cat to glove
                    #print("ann glove len %s"%len(ann_glove_list))
                    #print("noun len %s"%len(noun_glove_list))
                    cos_sim_list = [cosine_similarity(ann_glove, noun_glove)  # caculate all ann_cat glove and noun_glove sim,and >0.4 is cxt box
                                    for ann_glove in ann_glove_list
                                    for noun_glove in noun_glove_list]
                    #print("100cos_sim_list%s"%(cos_sim_list,))
                    # print(CAT_ID_TO_NAME[ann['category_id']], cos_sim_list)
                    max_cos_sim = max(cos_sim_list, default=0.)
                    #print("102max_cos_sim%s"%max_cos_sim)
                    #print("103max_cos_sim%s"%(max_cos_sim.shape,))
                    if max_cos_sim > 0.4:  # threshold
                        ann_box = xywh_to_xyxy(ann['bbox'])
                        if calculate_iou(ann_box, gt_box) > 0.9:  # if iou>0.9 is gt,else is 
                            gt_hit = True
                        else:
                            ctx_list.append({'box': ann_box, 'cat_id': ann['category_id']})
                if not gt_hit:
                    gt_miss_num += 1
                if not ctx_list:
                    empty_num += 1
                exp_to_ctx[sent_id] = {'gt': {'box': gt_box, 'cat_id': gt_cat}, 'ctx': ctx_list} # build cxt
                coco_box_num_list.append(len(refer.imgToAnns[image_id]))  #all ann box
                ctx_box_num_list.append(len(ctx_list) + 1)
        print('128GT miss: {} out of {}'.format(gt_miss_num, sent_num)) 
        print('empty ctx: {} out of {}'.format(empty_num, sent_num))
        print('COCO box per sentence: {:.3f}'.format(sum(coco_box_num_list) / len(coco_box_num_list)))
        print('ctx box per sentence: {:.3f}'.format(sum(ctx_box_num_list) / len(ctx_box_num_list)))
        ctxdb[split] = exp_to_ctx
    # Save results
        save_path1  = '/home_data/wj/ref_nms/cache/ref_proposal_info{}.json'.format(dataset_splitby)
        ctx_info = [split,'128GT miss: {} out of {}'.format(gt_miss_num, sent_num),'empty ctx: {} out of {}'.format(empty_num, sent_num),'COCO box per sentence: {:.3f}'.format(sum(coco_box_num_list) / len(coco_box_num_list)),'ctx box per sentence: {:.3f}'.format(sum(ctx_box_num_list) / len(ctx_box_num_list))]
        with open(save_path1, 'a') as f:
            json.dump(ctx_info, f)
        ctxdb[split] = exp_to_ctx
    # Save results
    save_path = '/home_data/wj/ref_nms/cache/std_ctxdb_{}.json'.format(dataset_splitby)
    print('saving ctxdb to {}'.format(save_path))
    with open(save_path, 'w') as f:
        json.dump(ctxdb, f)


def main():
    print('building ctxdb...')
    #for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
    for dataset, split_by in [ ('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_ctxdb(dataset, split_by)
    print()


main()
