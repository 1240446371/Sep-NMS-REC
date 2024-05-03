import numpy as np
import pytest
import torch
from PIL import Image
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-4])
print("9path%s"%root_path)
from clipmodel import clip


@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device, jit=True)
    py_model, _ = clip.load(model_name, device=device, jit=False)
    
    #img_test = Image.open("CLIP.png").shape
    #print("16img shape%s"%img_test.shape)
    
    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    print("20img shape%s"%image.shape)
    print("21text %s"%text)
    print("22text shape%s"%text.shape)
    
    with torch.no_grad():
        logits_per_image, _ = jit_model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)



def main():
    test_consistency('ViT-B/16')