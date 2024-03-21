# FashionFail

The official repository of _"FashionFail: Addressing Failure Cases in Fashion Object Detection and Segmentation"_.

**TL;DR**: Dataset: [![Generic badge](https://img.shields.io/badge/🤗-Datasets-blue.svg)](https://huggingface.co/datasets/rizavelioglu/fashionfail), Models: [![Generic badge](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/rizavelioglu/segmentation), Demo: [![Generic badge](https://img.shields.io/badge/🤗-Spaces-blue.svg)](https://huggingface.co/spaces/rizavelioglu/fashion-segmentation)


## Install

## Usage

### FashionFail Dataset

Check out [this file](./docs/00_dataset_creation.md) to get detailed information on how the dataset was generated, _i.e._
scraping, filtering, annotating and quality review.

The dataset _(annotations and image URLs)_ is available at [![Generic badge](https://img.shields.io/badge/🤗-Datasets-blue.svg)](https://huggingface.co/datasets/rizavelioglu/fashionfail).
Execute the following to download the dataset (which downloads the annotation files and images):
```bash
python fashionfail/data/make_dataset.py \
--save_dir "dir/to/save" \  # [optional] default: "~/.cache/fashionfail/"
```

### Training



### Inference

### Trained Models
- `Facere`:
    - *facere_1-v5-epoch=124-val_loss_sum=0.63.ckpt*
    - *facere_test2-epoch=125-val_loss_sum=0.55.ckpt*
- `Facere+`:
    - *facere_plus-epoch=06-val_loss_sum=0.13.ckpt*
    - *facere_plus-epoch=111-val_loss_sum=0.07.ckpt*
- `Attribute Mask R-CNN`:
  - *fashionpedia-spinenet-143/model.ckpt*
  - *fashionpedia-r50-fpn/model.ckpt*
- `Fashionformer`:
  - *fashionformer_swin_b_3x.pth*
  - *fashionformer_r50_3x.pth*


---
### Project Structure
The following project/directory structure is adopted:
[Cookiecutter Data Science by DrivenData](https://drivendata.github.io/cookiecutter-data-science/).
