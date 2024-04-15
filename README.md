# FashionFail

The official repository of _"FashionFail: Addressing Failure Cases in Fashion Object Detection and Segmentation"_.

**TL;DR**:
[![Generic badge][logo-hf_datasets]][ff-hf_datasets]
[![Generic badge][logo-hf_models]][ff-hf_models]
[![Generic badge][logo-hf_spaces]][ff-hf_spaces]
[![arXiv][logo-ff-paper]][ff-paper]


## Install
[todo]
```bash
pip install fashionfail       # basic usage, e.g. downloading dataset
pip install fashionfail[dev]  # full development, e.g. training, inference, evaluation
```


## Usage
> We offer a comprehensive explanation of each step, including dataset creation, training, and evaluation. You can find
detailed documentation [here.](references/00_table_of_content.md)

### FashionFail Dataset
The dataset _(annotations and image URLs)_ is available at [![Generic badge][logo-hf_datasets]][ff-hf_datasets].
Execute the following to download the dataset (which downloads the annotation files and images for each `train`, `val`,
and `test` splits):
```bash
python fashionfail/data/make_dataset.py \
--save_dir "dir/to/save" \  # [optional] default: "~/.cache/fashionfail/"
```


### FashionFail Models
Trained models are available at [![Generic badge][logo-hf_models]][ff-hf_models], where you can download the models.
To run inference using models, execute:

```bash
python fashionfail/models/predict_models.py \
--model_path "<PATH_TO_ONNX>" \
--out_dir "<OUTPUT_DIR>" \        # predictions are stored here
--image_dir "<IMAGES_DIR>"        # image dir to run inference for
```


Alternatively, you can check out the demo in a browser at [![Generic badge][logo-hf_spaces]][ff-hf_spaces].
One could also use the API of the demo to run inference:
```bash
python fashiofail/models/predict_hf_space.py \
--path_to_img "<PATH_TO_IMAGE>.jpg"
```



### Training

> Please see **_detailed explanation_** of training both `Facere` and `Facere+` models [here.](references/03_training.md)

> And see the inference for `Attribute Mask R-CNN` and `Fashionformer` [here.](references/05_evaluation.md)

---
### Project Structure
The following project/directory structure is adopted:
[Cookiecutter Data Science by DrivenData][cookiecutter].

```
.
├── LICENSE
├── notebooks/          # Jupyter Notebooks
├── pyproject.toml      # The requirements file for reproducing the environment
├── README.md
├── references/         # Explanatory materials
├── src/
│   └── data/           # Scripts to download or generate data
│   └── features/       # Scripts to turn raw data into features
│   └── models/         # Scripts to train models, use trained models to make predictions
│   └── visualization/  # Scripts to create exploratory and results oriented visualizations
```

### License

### Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation:
```
@inproceedings{velioglu2024fashionfail,
  author    = {Velioglu, Riza and Chan, Robin and Hammer, Barbara},
  title     = {FashionFail: Addressing Failure Cases in Fashion Object Detection and Segmentation},
  journal   = {IJCNN},
  eprint    = {2404.08582},
  year      = {2024},
}
```

[logo-hf_datasets]: https://img.shields.io/badge/🤗-Datasets-blue.svg?style=plastic
[logo-hf_models]: https://img.shields.io/badge/🤗-Models-blue.svg?style=plastic
[logo-hf_spaces]: https://img.shields.io/badge/🤗-Demo-blue.svg?style=plastic
[logo-ff-paper]: https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=plastic
[ff-hf_datasets]: https://huggingface.co/datasets/rizavelioglu/fashionfail
[ff-hf_models]: https://huggingface.co/rizavelioglu/fashionfail
[ff-hf_spaces]: https://huggingface.co/spaces/rizavelioglu/fashionfail
[ff-paper]: https://arxiv.org/abs/2404.08582
[cookiecutter]: https://drivendata.github.io/cookiecutter-data-science/
