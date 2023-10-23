# Shatter and Gather: Learning Referring Image Segmentation with Text Supervision

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2308.15512)  [![Badge](https://img.shields.io/badge/ProjectPage-SaG-Green.svg?)](https://southflame.github.io/sag/)

![plot](./teaser.jpg)

This repository contains the official source code for our paper:
>[Shatter and Gather: Learning Referring Image Segmentation with Text Supervision](https://arxiv.org/abs/2308.15512)  
> [Dongwon Kim<sup>1</sup>](https://kdwonn.github.io/),
> [Namyup Kim<sup>1</sup>](https://southflame.github.io/), 
> [Cuiling Lan<sup>2</sup>](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=en), and
> [Suha Kwak<sup>1</sup>](https://suhakwak.github.io/) <br>
> <sup>1</sup>POSTECH CSE, <sup>2</sup>Microsoft Research Asia<br>
> ICCV, Paris, 2023.

# Dataset setup
## Setting
- [Download](http://mscoco.org/dataset/#download) the MS COCO images are under `data/coco/images/train2014/`
- [Download](http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz) the ReferItGame data are under `data/referit/images` and `data/referit/mask`
- Download [TF-resnet](https://github.com/chenxi116/TF-resnet) and [TF-deeplab](https://github.com/chenxi116/TF-deeplab) under `external` folder. Then strictly foll 
- Download [refer](https://github.com/chenxi116/refer) under `external`. Then strictly follow the [Setup](https://github.com/chenxi116/refer#setup) and [Download](https://github.com/chenxi116/refer#download) section. Also put the `refer` folder in `PYTHONPATH`
- Download the [MS COCO API](https://github.com/pdollar/coco) also under `external` (i.e. `external/coco/PythonAPI/pycocotools`)

## Data preparation
```
python build_batches.py -d Gref -t train 
python build_batches.py -d Gref -t val 
python build_batches.py -d unc -t train 
python build_batches.py -d unc -t val 
python build_batches.py -d unc -t testA 
python build_batches.py -d unc -t testB 
python build_batches.py -d unc+ -t train 
python build_batches.py -d unc+ -t val 
python build_batches.py -d unc+ -t testA 
python build_batches.py -d unc+ -t testB
```

## Final `./data` directory structure
```
./data              
├─ refcoco   
│   ├─ Gref
│   │   ├─ train_batch
│   │   │   ├─ Gref_train_0.npz
│   │   │   ├─ Gref_train_1.npz
│   │   │   └─ ...
│   │   ├─ train_image
│   │   ├─ train_label 
│   │   ├─ val_batch
│   │   ├─ val_image
│   │   └─ val_label
│   ├─ unc
│   │   └─ ...
│   └─ unc+
│       └─ ...
├─ phrasecut
│   └─ images
│      ├─ refer_train_ris.json
│      ├─ refer_val_ris.json
│      └─  refer_test_ris.json
├─ Gref_emb.npy
├─ referit_emb.npy
├─ vocabulary_Gref.txt
└─ vocabulary_referit.txt
```

# Environment setup

* Python 3.10.9
* PyTorch 1.13.1+cu117

Instructions:
```shell
conda create -n sag python=3.10 -y
conda activate sag
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu11
pip install einops tqdm wandb transformers
pip install matplotlib timm opencv-python
```

# Train & eval

```shell
sh ./train_eval_gref.sh # Gref
sh ./train_eval_unc.sh # UNC
sh ./train_eval_unc+.sh # UNC+
```
## Acknowledgement
Parts of our codes are adopted from the following repositories.
* https://github.com/lucidrains/perceiver-pytorch
* https://github.com/kdwonn/DivE

Dataset Setup instruction is from [TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public) repository.
