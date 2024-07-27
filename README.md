# FashionFail

The official repository of _"FashionFail: Addressing Failure Cases in Fashion Object Detection and Segmentation"_.

**TL;DR**:
[![Generic badge][logo-hf_datasets]][ff-hf_datasets]
[![Generic badge][logo-hf_models]][ff-hf_models]
[![Generic badge][logo-hf_spaces]][ff-hf_spaces]
[![arXiv][logo-ff-paper]][ff-paper]


## Install
Create a new environment named `ff` (short for FashionFail):
```bash
conda create -n ff python=3.9
conda activate ff
```

Then, clone the repository and install the package via:
```bash
git clone https://github.com/rizavelioglu/fashionfail.git
cd fashionfail
pip install -e .[dev]  # full development, e.g. training, inference, evaluation
```

**_Note:_** If you just want to construct the FashionFail dataset, you could install the package via
the following instead, which avoids downloading heavy packages:
```bash
pip install -e .       # basic usage, i.e. downloading dataset
```

## Usage
> We offer a comprehensive explanation of each step, including dataset creation, training, and evaluation. You can find
detailed documentation at [references/README.md](references/README.md)

### FashionFail Dataset
The dataset _(annotations and image URLs)_ is available at [![Generic badge][logo-hf_datasets]][ff-hf_datasets].
Execute the following to download the dataset (which downloads the annotation files and images for each `train`, `val`,
and `test` splits):
```bash
python fashionfail/data/make_dataset.py \
--save_dir "dir/to/save" \  # [optional] default: "~/.cache/fashionfail/"
```
An optional argument `--save_dir` can be set to construct the dataset in the preferred directory, but it is not
recommended to alter the default location, as the training script expects the data to be at a specific location.


### FashionFail Models
Trained models are available at [![Generic badge][logo-hf_models]][ff-hf_models].
To run inference using models execute the following, which downloads the model if necessary:
```bash
python fashionfail/models/predict_models.py \
--model_name "facere" \                       # or "facere+"
--out_dir "<OUTPUT_DIR>" \                    # predictions are stored here
--image_dir "<IMAGES_DIR>"                    # image dir to run inference for
```

Alternatively, you can check out the demo in a browser at [![Generic badge][logo-hf_spaces]][ff-hf_spaces] and run
inference using the user interface.



### Training

> Please refer **_detailed explanation_** of training both `Facere` and `Facere+` models at [references/03_training.md](references/03_training.md)

> And see the inference for `Attribute Mask R-CNN` and `Fashionformer` at [references/04_inference.md](references/04_inference.md)

---
### Project Structure
The following project/directory structure is adopted:
[Cookiecutter Data Science-v1 by DrivenData][cookiecutter].

```
.
├── LICENSE
├── notebooks/          # Jupyter Notebooks
├── pyproject.toml      # The requirements file for reproducing the environment
├── README.md
├── references/         # Explanatory materials
├── src/
│   └── data/           # Scripts to download or construct dataset
│   └── features/       # Scripts to turn raw data into features
│   └── models/         # Scripts to train models, use trained models to make predictions
│   └── visualization/  # Scripts to create exploratory and results oriented visualizations
```

### License
**_TL;DR_**: Not available for commercial use, unless the FULL source code is open-sourced!\
This project is intended solely for academic research. No commercial benefits are derived from it.\
The code, datasets, and models are published under the [Server Side Public License (SSPL)](LICENSE).

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
[cookiecutter]: https://cookiecutter-data-science.drivendata.org/v1/
