
We follow the settings of ：Official codebase for AAAI 2021 paper ["Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding"](https://arxiv.org/abs/2009.01449).

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

# Ctx-relatedness 
We follow the code of Ref-NMS: ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding" [Ref-NMS](https://github.com/ChopinSharp/ref-nms) In their 
```
refnms/lib/predictor.py"
```
# Ref-relatedness
“lib/my_sep_qkad_predictor.py” incoporate the  Ctx-relatedness and  Ref-relatedness module
```
refnms/lib/predictor.py
```

# other codings：
containing the training and test part， you can follow the RefNMS：
ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding" (github.com)

## Pretrained Models
We provide pre-trained model weights as long as the corresponding **MAttNet-style detection file** (note the MattNet-style detection files can be directly used to evaluate downstream REG task performance). With these files, one can easily reproduce our reported results.

[[Google Drive]](https://drive.google.com/drive/folders/1BPqWW0LrAEBFna7b-ORF2TcrY7K_DDvM?usp=sharing) [[Baidu Disk]](https://pan.baidu.com/s/1G4k7APKSUs-_5StXoYaNrA) (extraction code: 5a9r)

 
