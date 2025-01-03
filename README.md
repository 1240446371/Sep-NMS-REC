# The official code for the paper：Sep-NMS: Unlocking the Aptitude of Two-Stage Referring Expression Comprehension
We follow the settings of ：Official codebase for AAAI 2021 paper ["Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding"](https://arxiv.org/abs/2009.01449). 
Please ensure that the parameter settings in the code adhere to the experimental details provided in the Sep-NMS paper.

## Prerequisites
The following dependencies should be enough. See [environment.yml](environment.yml) for complete environment settings.
- python 3.7.6
- pytorch 1.1.0
- torchvision 0.3.0
- tensorboard 2.1.0
- spacy 2.2.3

## Data Preparation
Follow instructions in `data/README.md` to setup `data` directory. 
## Initial Filtering
This part is identical to the content in Ref-NMS. So we follow the code of Ref-NMS: ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding" [Ref-NMS](https://github.com/ChopinSharp/ref-nms) 

# CLIP† relatedness 
1、 CLIP† denotes a variant of the CLIP model that has been fine-tuned on the MS COCO dataset. We utilze the ITM module in GR-GAN paper. The original code is following:  BoO-18/GR-GAN: GRADUAL REFINEMENT TEXT-TO-IMAGE GENERATION (github.com) [BoO-18/GR-GAN](https://github.com/BoO-18/GR-GAN)

2、 The following code corresponds to the  CLIP† relatedness mentioned in our paper. CLIP† relatedness aims to filter referent and context proposals, the output is the simliarity score. 
```
tools/ybclip_ann_sent.py
```

# Ctx-relatedness &&  Ref-relatedness
The Ctx-Relatedness module is identical to the  [Ref-NMS](https://github.com/ChopinSharp/ref-nms) model. The original code for this component can be found in the following directory of the Ref-NMS codebase.
```
/lib/predictor.py"
```

In our codebase, the architectures for both the Ctx-Relatedness and Ref-Relatedness models can be found in the following directory:
```
“lib/my_sep_qkad_predictor.py” 
```
which incoporates the  Ctx-relatedness and  Ref-relatedness module

# Train：
Train Ctx-relatedness &&  Ref-relatedness with binary XE loss:
```
tools/my_qkad_train2.py
```

# other codings：
The tools and lib directories contain various experiments and attempts of our method under different settings and parameter configurations.

For the test part， you can follow the RefNMS：
 [Ref-NMS](https://github.com/ChopinSharp/ref-nms) 

## Pretrained Models
We provide pre-trained model weights as long as the corresponding **MAttNet-style detection file** (note the MattNet-style detection files can be directly used to evaluate downstream REG task performance). With these files, one can easily reproduce our reported results.

[[Google Drive]](https://drive.google.com/drive/folders/1BPqWW0LrAEBFna7b-ORF2TcrY7K_DDvM?usp=sharing) [[Baidu Disk]](https://pan.baidu.com/s/1G4k7APKSUs-_5StXoYaNrA) (extraction code: 5a9r)

 
