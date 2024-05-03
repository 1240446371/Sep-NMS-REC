# -*- coding:utf-8 -*-
import numpy as np
import pytest
import torch
from PIL import Image
import os
import sys
#root_path = os.path.abspath(__file__)
#root_path = '/'.join(root_path.split('/')[:-2])
import clip
import cv2
import numpy as np

@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device, jit=True)
    py_model, _ = clip.load(model_name, device=device, jit=False)
    
    #img_test = Image.open("CLIP.png").shape
    #print("16img shape%s"%img_test.shape)
    img_open = Image.open("/home/wj/code/ref_nms_main/tools/CLIP.png")
    img_open1 = np.array(img_open)
    #print("21img_open%s"%(img_open1.shape,))
    image_open = transform(Image.open("/home/wj/code/ref_nms_main/tools/CLIP.png")).unsqueeze(0).to(device)
    #print("25image_open%s"%image_open)
    img_read = cv2.imread("/home/wj/code/ref_nms_main/tools/CLIP.png")
    #print("23img_read%s"%(img_read.shape,))
    im_read1 = Image.fromarray(img_read)
    image_read = transform(im_read1).unsqueeze(0).to(device)
    #print("30image read %s"%image_read)
    #text = clip.tokenize(["diagram", "a dog", "a cat",""," ","unk"]).to(device)
    #text = clip.tokenize([ "this","is","a","diagram"]).to(device)
    text = clip.tokenize([ "this is a dog","this is a cat","this is a diagram"]).to(device)
    #print("26image%s"%image)
    #print("27img shape%s"%(image.shape,))
    #print("28text %s"%text)
    #print("29text shape%s"%(text.shape,))
    
    with torch.no_grad():
        logits_per_image, _ = jit_model(image_open, text)
        print("41logits_per_image%s"%logits_per_image)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print("34logits_per_image%s"%logits_per_image)
        print("35jit_probs%s"%jit_probs)
        #logits_per_image, _ = jit_model(image_read, text)
        #jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print("44jit_probs%s"%jit_probs)
        #logits_per_image, _ = py_model(image_read, text)
        #py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print("43py_probs%s"%py_probs)

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)



def main():
    test_consistency('ViT-B/16')
    
if __name__ == '__main__':
    main()    