import json
import cv2
import numpy as np
import spacy
from tqdm import tqdm
import os
import sys
from time import *
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torchvision.transforms as transforms
import torch.nn.functional as F
from lib.refer import REFER
from utils.constants import CAT_ID_TO_NAME, EVAL_SPLITS_DICT
from utils.misc import xywh_to_xyxy, calculate_iou
from io import BytesIO as StringIO
import clip
import torch   
from PIL import Image
POS_OF_INTEREST = {'NOUN', 'NUM', 'PRON', 'PROPN'}
import os.path as osp
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from ybclip import CLIP_MODEL

   
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



# image box and token and score
def draw_attention(data, loader, n):
    image_root = '/COCO/'

    plt.figure()
    ax = plt.gca()
    I = io.imread(osp.join(image_root, data['images']['file_name']))
    ax.imshow(I)

    bbox = loader.Anns[data['ref'][n]['ann_id']]['box']
    box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='#1f953b', linewidth=2)
    ax.add_patch(box_plot)

    plt.axis('off')
    plt.show()
    plt.cla()
    plt.close("all")

def cosine_similarity(feat_a, feat_b):
    return np.sum(feat_a * feat_b) / np.sqrt(np.sum(feat_a * feat_a) * np.sum(feat_b * feat_b))


def sent_loss_clip(cnn_code, rnn_code, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
  
        # --> seq_len x batch_size x nef
        print(cnn_code.size())
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)
        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * 10.0
        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
 
        # print(loss1, loss0)
        return scores0
        

def build_ctxdb(dataset, split_by):

    device_cuda=torch.device('cuda',1)
    
    clip_model = CLIP_MODEL(512, 'CLIP_FT')
    state_dict = \
        torch.load("/home/wj/clip_model149.pth", map_location=lambda storage, loc: storage)
    clip_model.load_state_dict(state_dict)
    for p in clip_model.parameters():
        p.requires_grad = False
    #print('Load clip encoder from:', cfg.TRAIN.NET_E)
    #clip_model.cuda()
    clip_model.float()
    clip_model.to(device_cuda)
    clip_model.eval()
    
    #jit_model, transform = clip.load('ViT-B/16', device=device_cuda,jit=False)
    dataset_splitby = '{}_{}'.format(dataset, split_by)
    # Load refer
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset, split_by)
    IMAGE_DIR ="/data1/qyx/VQA-X/Images/train2014/"
    # Load GloVe feature
    #glove_dict = load_glove_feats()
    # Construct COCO category GloVe feature

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
    for split in ( EVAL_SPLITS_DICT[dataset_splitby]+['train']):
        
        exp_to_ctx = {}
        gt_miss_num, empty_num, sent_num = 0, 0, 0
        coco_box_num_list, ctx_box_num_list = [], []
        ref_ids = refer.getRefIds(split=split)
        print("69 split %s"%split)
        print("70 ref_ids %s"%len(ref_ids))
        num_ref =0
        time_begin =time()
        for ref_id in tqdm(ref_ids, ascii=True, desc=split):
            time_begin =time()
            ref = refer.Refs[ref_id]
            #print("74ref%s"%ref)
            image_id = ref['image_id']
            #print("76imgid%s"%image_id)
            # read img 
            img_pth=os.path.join(IMAGE_DIR,'COCO_train2014_'+str(image_id).zfill(12)+'.jpg')
            im = cv2.imread(img_pth)
            # draw img
            gt_box = xywh_to_xyxy(refer.Anns[ref['ann_id']]['bbox'])
            #print("82gt_box%s"%(gt_box,))
            gt_cat = refer.Anns[ref['ann_id']]['category_id']  
            for sent in ref['sentences']:
                #print("66 sentence%s"% len(ref['sentences']))
                #print("67sent %s"%sent)
                #print("94sent['sent']%s"%sent['sent'])
                sent_num += 1
                sent_id = sent['sent_id']
                #doc = nlp(sent['sent'])
                sentence = sent['sent']
                #noun_tokens = [token.text for token in doc if token.pos_ in POS_OF_INTEREST]
                print("125sentence %s"%sentence)
                #print("74tokentext%s"%token.text)
                #print("65tokentext%s"%token.text)
                #print('67SENT', sent['sent'])    
                #print('80NOUN TOKENS', noun_tokens)
                #noun_glove_list = [np.array(glove_dict[t], dtype=np.float32) for t in noun_tokens if t in glove_dict]  # get noun_token's glove feats
                gt_hit = False
                ctx_list = []
#96ann{'segmentation': [[478.36, 78.27, 449.22, 109.65, 432.42, 179.13, 423.45, 231.79, 430.17, 276.62, 423.45, 291.19, 425.69, 355.06, 436.9, 353.94, #439.14, 367.39, 452.59, 366.27, 452.59, 372.99, 435.78, 384.2, 425.69, 392.04, 416.73, 444.71, 416.73, 506.34, 438.02, 525.39, 451.47, 513.06, 459.31, #463.76, 463.79, 416.69, 470.52, 393.16, 477.24, 387.56, 479.48, 340.49, 480.0, 337.29, 480.0, 291.35, 478.6, 217.39, 477.48, 175.93]], 'area': 18847
#.397299999997, 'iscrowd': 0, 'image_id': 579382, 'bbox': [416.73, 78.27, 63.27, 447.12], 'category_id': 1, 'id': 201995}
                clip_score = []
                #clip_score_array = np.zeros[len(noun_tokens),len(refer.imgToAnns[image_id])]
                ann_box_list = []
                list1 = []
                # draw 
                """plt.figure()
                ax = plt.gca()
                I = io.imread(img_pth)
                ax.imshow(I)
                box_plot = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0]+1,gt_box[3]-gt_box[1]+1, fill=False, edgecolor='green', linewidth=3)
                ax.add_patch(box_plot)
                #ax.text(0, 0, str(sent['sent']), bbox={'facecolor':'green', 'alpha':0.5})
                plt.xticks([0], [str(sent['sent'])])"""
                for ann in refer.imgToAnns[image_id]:  # ann has many 7 annotations include cat_id 
                    ann_box = xywh_to_xyxy(ann['bbox'])
                    #print("125 annbox type %s"%type(ann_box)) # tuple
                    #print("126 annbox  %s"%(ann_box,))
                    ann_box_list.append(ann_box)
                    x1,y1,x2,y2 = int(ann_box[0]),int(ann_box[1]), int(ann_box[2])+1,int(ann_box[3])+1
                    im_box = im[y1:y2,x1:x2]
                    im_box=Image.fromarray(im_box)
                    im_box = im_box.convert('RGB')
                    norm = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    im_box = norm(im_box).unsqueeze(dim=0).to(device_cuda)
                    #print("176im_box%s"%(im_box.shape,))
                    #print("box shape %s %s , im_box %s"%(x2-x1,y2-y1,im_box0.shape))
                    #roi_input = im_box
                    _, roi_input = clip_model.image_encode(im_box)
                    roi_input = roi_input.to(device_cuda)
                    
                    sentence0 = clip.tokenize(sentence).to(device_cuda)
                    _,re = clip_model.sent_encode(sentence0)
                    re = re.to(device_cuda)
                    #print("!!!!!!!!!!!!!!!!!!!!!!!!!", roi_input.size(), re.size())
                    logits_per_image=  sent_loss_clip(roi_input,re).to(device_cuda)
                    
                    #logits_per_image, per_text = clip_model(roi_input, re)
                    #roi_input = transform(im_box).unsqueeze(0).to(device_cuda) 
                    #re = clip.tokenize(sentence).to(device_cuda)
                    #logit_scale,box_feature,token_feat,logits_per_image, per_text = jit_model(roi_input, re)
                    #logit_scale,box_feature,token_feat,logits_per_image, per_text = jit_model(roi_input, re)
                    #print("182score%s"%logits_per_image)
                    clip_cos_list=[]
                    logits_per_image = torch.squeeze(logits_per_image)
                    list1= logits_per_image.tolist()
                    clip_score.append(list1)
                    
                clip_score_array = np.asarray(clip_score)
                clip_score_tensor = (torch.from_numpy(clip_score_array))
                clip_soft = F.softmax(clip_score_tensor,dim =0).t()
                print("157clip_soft%s"%clip_soft)
                clip_soft_to_ = clip_soft
                clip_soft_to_array = clip_soft_to_.detach().numpy()
                
                i =0
                #ax.text(draw_box[2], draw_box[3],str(noun_tokens))
                for item in clip_soft:
                    """draw_box = ann_box_list[i]
                    box_plot = Rectangle((draw_box[0], draw_box[1]), draw_box[2]-draw_box[0]+1,draw_box[3]-draw_box[1]+1, fill=False, edgecolor='blue', linewidth=1)
                    ax.add_patch(box_plot)
                    ax.text(draw_box[0], draw_box[1], '{:.1f}'.format(item.item()), bbox={'facecolor':'white', 'alpha':0.5})
                    #ax.text(draw_box[0], draw_box[1], '{:.1f}'.format(item.item()))"""
                    
                    if item>=0.03:
                        # draw
                        if calculate_iou(ann_box_list[i], gt_box) > 0.9:  # if iou>0.9 is gt,else is 
                            gt_hit = True
                        else:
                            ctx_list.append({'box': ann_box, 'cat_id': ann['category_id']})
                    i+=1
                """result_path = '/data1/wj/ref_nms/result/resultsent/{}'.format(str(sent['sent'])+ 'jpg')
                plt.savefig(result_path)
                print("200save ok")
                ax.set_axis_off()
                ax.set_yticks([])
                ax.set_xticks([])"""

                    # last detroy the plt
                           
                if not gt_hit:
                    gt_miss_num += 1
                if not ctx_list:
                    empty_num += 1
                exp_to_ctx[sent_id] = {'gt': {'box': gt_box, 'cat_id': gt_cat}, 'ctx': ctx_list} # build cxt
                coco_box_num_list.append(len(refer.imgToAnns[image_id]))  #all ann box
                ctx_box_num_list.append(len(ctx_list) + 1)
            num_ref += 1
            time_end =time()
            cost_time = time_end - time_begin
            print("num %s is ok time is %s"%(num_ref,cost_time))
        print('128GT miss: {} out of {}'.format(gt_miss_num, sent_num)) 
        print('empty ctx: {} out of {}'.format(empty_num, sent_num))
        print('COCO box per sentence: {:.3f}'.format(sum(coco_box_num_list) / len(coco_box_num_list)))
        print('ctx box per sentence: {:.3f}'.format(sum(ctx_box_num_list) / len(ctx_box_num_list)))
        #save_path1  = '/data1/wj/ref_nms/cache/ybclip_ctx_04seninfo{}.json'.format(dataset_splitby)
        save_path1  = '/home_data/wj/ref_nms/cache/ybclip_ctx_03seninfo{}.json'.format(dataset_splitby)
        ctx_info = [split,'128GT miss: {} out of {}'.format(gt_miss_num, sent_num),'empty ctx: {} out of {}'.format(empty_num, sent_num),'COCO box per sentence: {:.3f}'.format(sum(coco_box_num_list) / len(coco_box_num_list)),'ctx box per sentence: {:.3f}'.format(sum(ctx_box_num_list) / len(ctx_box_num_list))]
        with open(save_path1, 'a') as f:
            json.dump(ctx_info, f)
        ctxdb[split] = exp_to_ctx
    # Save results
    #save_path = '/data1/wj/ref_nms/cache/ybclip_ctxdb_ann_04sen{}.json'.format(dataset_splitby)
    save_path = '/home_data/wj/ref_nms/cache/ybclip_ctxdb_ann_03sen{}.json'.format(dataset_splitby)
    print('saving ctxdb to {}'.format(save_path))
    with open(save_path, 'w') as f:
        json.dump(ctxdb, f)
  
  
def main():
    print('building ctxdb...')
    #for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
    for dataset, split_by in [('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_ctxdb(dataset, split_by)
    print()


main()
