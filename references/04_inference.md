# Inference
We compare our `Facere` models with the state-of-the-art Attribute Mask R-CNN `(amrcnn)` and FashionFormer `(fformer)`
models. Since their repositories have conflicting dependencies, we create separate virtual environments for each of
them. Here are all the models we used:

| Model name        | Description                                                                 |   Backbone   | `FashionFail-train` data | download                                                |
|-------------------|-----------------------------------------------------------------------------|:------------:|:------------------------:|:--------------------------------------------------------|
| `amrcnn-spine`    | Attribute Mask-RCNN model released with [Fashionpedia paper][paper_amrcnn]. | SpineNet-143 |            x             | [ckpt][amrcnn_spine_ckpt] \| [config][amrcnn_spine_cfg] |
| `fformer-swin`    | Fashionformer model released by [Fashionformer paper][paper_fformer].       |  Swin-base   |            x             | [pth][models_fformer]                                   |
| `amrcnn-r50-fpn`  | Attribute Mask-RCNN model released with [Fashionpedia paper][paper_amrcnn]. | ResNet50-FPN |            x             | [ckpt][amrcnn_r50_ckpt] \| [config][amrcnn_r50_cfg]     |
| `fformer-r50-fpn` | Fashionformer model released by [Fashionformer paper][paper_fformer].       | ResNet50-FPN |            x             | [pth][models_fformer]                                   |
| `facere`          | Mask R-CNN based model trained on `Fashionpedia-train`.                     | ResNet50-FPN |            x             | [onnx][facere_onnx]                                     |
| `facere+`         | `facere` model finetuned on `FashionFail-train`.                            | ResNet50-FPN |            ✔             | [onnx][facere_plus_onnx]                                |


### 1. Inference on `Facere` models
(After training) Execute the following command to convert the trained model (`.ckpt`) to `.onnx` format:
```bash
python models/export_to_onnx.py \
--ckpt_path "model.ckpt" \
--onnx_path "model.onnx" \
--model_class "facere_base"  # either "facere_base" or "facere_plus"
```

Then, run inference using `ONNX Runtime` with:
```bash
python models/predict_models.py \
--model_name "facere_base" \  # either "facere_base" or "facere_plus"
--image_dir "path/to/images/to/run/inference/for/" \
--out_dir "path/to/where/predictions/will/be/saved/"
```

which saves all the predictions into a single compressed `.npz` file, which is storage-efficient.
The file has the following structure:
```python
{
    "image_file": str,        # image file name
    "boxes": numpy.ndarray,   # boxes in yxyx format (same as `amrcnn` model output)
    "classes": numpy.ndarray, # classes/categories in [1,n] for n classes
    "scores": numpy.ndarray,  # confidence scores of boxes in [0,1]
    "masks": list(dict),      # segmentation masks in encoded RLE format
}
```

Alternatively, see the inference code in [HuggingFace Spaces][hf_spaces_app].

### 2. Inference on Attribute Mask R-CNN [[paper]][paper_amrcnn] [[code]][code_amrcnn]
> Note on the repository: The whole repository is really complex and not easily editable, e.g. I couldn't run inference
> on GPUs, failed to convert the model to `.onnx` format, etc.
> Therefore, the following procedure is not optimal, but it works...

Create and activate the conda environment:
```bash
conda create -n amrcnn python=3.9
conda activate amrcnn
```

Install dependencies:
```bash
pip install tensorflow-gpu==2.11.0 Pillow==9.5.0 pyyaml opencv-python-headless tqdm pycocotools
```

Clone the repository, navigate to the `detection` directory and download the models:
```bash
cd /change/dir/to/fashionfail/repo/
git clone https://github.com/jangop/tpu.git
cd tpu
git checkout 85b65b6
cd models/official/detection
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz --output fashionpedia-spinenet-143.tar.gz
tar -xf fashionpedia-spinenet-143.tar.gz
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz
tar -xf fashionpedia-r50-fpn.tar.gz
```

The inference script expects a .zip file for the input images. Hence, zip the `FashionFail-test` data, for example:
```bash
cd ~/.cache/fashionfail/
tar -cvf ff_test.tar images/test/*
```

Finally, we can run inference with:
```bash
cd some_path/fashionfail/tpu/models/official/detection
python inference_fashion.py \
    --model="attribute_mask_rcnn" \
    --config_file="projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml" \
    --checkpoint_path="fashionpedia-spinenet-143/model.ckpt" \
    --label_map_file="projects/fashionpedia/dataset/fashionpedia_label_map.csv" \
    --output_html="out.html" --max_boxes_to_draw=8 --min_score_threshold=0.01 \
    --image_size="640" \
    --image_file_pattern="~/.cache/fashionfail/ff_test.tar" \
    --output_file="outputs/spinenet143-ff_test.npy"
```

The predictions file has the following structure:
```python
{
    'image_file': str,        # image file name
    'boxes': np.ndarray,      # boxes in yxyx format
    'classes': np.ndarray,    # classes/categories in [1,n] for n classes
    'scores': np.ndarray,     # confidence scores of boxes in [0,1]
    'attributes': np.ndarray, # attributes (not used in our evaluation)
    'masks': encoded_masks,   # segmentation masks in encoded RLE format
}

```

### 3. Inference on FashionFormer [[paper]][paper_fformer] [[code]][code_fformer]

Create and activate the conda environment:
```bash
conda create -n fformer python==3.8.13
conda activate fformer
```

Install dependencies:
```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
pip install -U openmim
mim install mmdet==2.18.0
mim install mmcv-full==1.3.18
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -U scikit-learn
pip install -U scikit-image
pip install torchmetrics
```

Clone the repository and create a new directory for the model weights:
```bash
cd /change/dir/to/fashionfail/repo/
git clone https://github.com/xushilin1/FashionFormer.git
mkdir FashionFormer/ckpts
```
Download the models manually from [OneDrive][models_fformer] and place them inside the newly created
`FashionFormer/ckpts` folder.

Then, run inference with:
```bash
python src/fashionfail/models/predict_fformer.py \
--model_path "./FashionFormer/ckpts/fashionformer_r50_3x.pth" \
--config_path  "./FashionFormer/configs/fashionformer/fashionpedia/fashionformer_r50_mlvl_feat_3x.py"\
--out_dir "path/to/where/predictions/will/be/saved/" \
--image_dir "./cache/fashionfail/images/test/" \
--dataset_name "ff_test" \
--score_threshold 0.05
```
which saves all the predictions into a single compressed `.npz` file, which is storage-efficient.
> Note: A `score_threshold=0.05` is applied to model predictions. This is because the `fformer` outputs a fixed
> number (100) of predictions for each input due to its Transformer architecture, resulting in many unconfident and
> mainly wrong predictions, which can lead to poor results. Therefore, this thresholding is applied to evaluate the
> model's performance fairly.

The predictions file has the following structure:
```python
{
    "image_file": str,        # image file name
    "boxes": numpy.ndarray,   # boxes in xyxy format
    "classes": numpy.ndarray, # classes/categories in [0,n-1] for n classes
    "scores": numpy.ndarray,  # confidence scores of boxes in [0,1]
    "masks": list(dict),      # segmentation masks in encoded RLE format
}
```



[paper_fformer]: https://arxiv.org/abs/2204.04654
[code_fformer]: https://github.com/xushilin1/FashionFormer
[models_fformer]: https://1drv.ms/u/s!Ai4mxaXd6lVBcAWlLG9x3sx8cKY?e=cBZdNy
[paper_amrcnn]: https://arxiv.org/abs/2004.12276
[code_amrcnn]: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/fashionpedia
[amrcnn_spine_ckpt]: https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz
[amrcnn_spine_cfg]: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml
[amrcnn_r50_ckpt]: https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz
[amrcnn_r50_cfg]: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/r50fpn_amrcnn.yaml
[hf_spaces_app]: https://huggingface.co/spaces/rizavelioglu/fashionfail/blob/main/app.py
[facere_onnx]: https://huggingface.co/rizavelioglu/fashionfail/resolve/main/facere_base.onnx?download=true
[facere_plus_onnx]: https://huggingface.co/rizavelioglu/fashionfail/resolve/main/facere_plus.onnx?download=true


---
<div style="display: flex; justify-content: space-between;">

   [Back](03_training.md)

   [Next: Evaluation](05_evaluation.md)

</div>
