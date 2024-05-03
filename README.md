
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

Run following script to setup `cache` directory:
```
sh scripts/prepare_data.sh
```
This should generate following files under `cache` directory:
- vocabulary file: `std_vocab_<dataset>_<split_by>.txt`
- selected GloVe feature: `std_glove_<dataset>_<split_by>.npy`
- referring expression database: `std_refdb_<dataset>_<split_by>.json`
- critical objects database: `std_ctxdb_<dataset>_<split_by>.json`


# CLIP† relatedness 
1、 “Ybclip” file is the modified clip model (CLIP†)we used in our paper. The original code is following:  BoO-18/GR-GAN: GRADUAL REFINEMENT TEXT-TO-IMAGE GENERATION (github.com)
2、 We use  "tools/ybclip_ann_sent.py" to filter referent and context proposals, the output is the simliarity score. 
This code corresponds to the  CLIP† relatedness mentioned in the paper.

# Ctx-relatedness 
We follow the code of Ref-NMS: ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding" (github.com)

# Ref-relatedness
“lib/my_sep_qkad_predictor.py” incoporate the  Ctx-relatedness and  Ref-relatedness module

# other codings：
containing the training and test part， you can follow the RefNMS：
ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding" (github.com)![image](https://github.com/1240446371/Sep-NMS-REC/assets/44427801/7103d919-d40b-45e3-a9b3-1855d7259156)

## Pretrained Models
We provide pre-trained model weights as long as the corresponding **MAttNet-style detection file** (note the MattNet-style detection files can be directly used to evaluate downstream REG task performance). With these files, one can easily reproduce our reported results.

[[Google Drive]](https://drive.google.com/drive/folders/1BPqWW0LrAEBFna7b-ORF2TcrY7K_DDvM?usp=sharing) [[Baidu Disk]](https://pan.baidu.com/s/1G4k7APKSUs-_5StXoYaNrA) (extraction code: 5a9r)

 
