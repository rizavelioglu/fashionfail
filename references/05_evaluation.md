# Evaluation

#### Trained Models
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

We compare our `Facere` models with the state-of-the-art `Attribute Mask R-CNN` (amrcnn) and `FashionFormer` (fformer)
models. Since their repositories have conflicting dependencies, we create separate virtual environments for each of
them.

### Attribute Mask R-CNN [[paper]][paper_amrcnn] [[code]][code_amrcnn]
> Note on the repository: The whole repository is really complex and not easily editable, e.g. I couldn't run inference
> on GPUs. Therefore, the following procedure is not optimal, but it works...

Create and activate the conda environment:
```bash
conda create -n amrcnn python=3.9
conda activate amrcnn
```

Install dependencies:
```bash
pip install tensorflow-gpu==2.11.0 Pillow==9.5.0 pyyaml opencv-python-headless tqdm pycocotools
```

Navigate to the `detection` directory and download the models:
```bash
cd tpu/models/official/detection
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz --output fashionpedia-spinenet-143.tar.gz
tar -xf fashionpedia-spinenet-143.tar.gz
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz
tar -xf fashionpedia-r50-fpn.tar.gz
```

The inference script expects a .zip file for the input images. Hence, zip the `FashionFail-test` data:
```bash
cd ~/.cache/fashionfail/
tar -cvf ff_train.tar images/test/*
```

Finally, we can run inference with:
```bash
cd path/to/tpu/models/official/detection
python inference_fashion.py \
    --model="attribute_mask_rcnn" \
    --config_file="projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml" \
    --checkpoint_path="fashionpedia-spinenet-143/model.ckpt" \
    --label_map_file="projects/fashionpedia/dataset/fashionpedia_label_map.csv" \
    --output_html="out.html" --max_boxes_to_draw=8 --min_score_threshold=0.01 \
    --image_size="640" \
    --image_file_pattern=".../fashionfail/data-v2/ff_test.tar" \
    --output_file="outputs/spinenet143-ff_test.npy"
```

### FashionFormer [[paper]][paper_fformer] [[code]][code_fformer]



[paper_fformer]: https://arxiv.org/abs/2204.04654
[code_fformer]: https://github.com/xushilin1/FashionFormer
[paper_amrcnn]: https://arxiv.org/abs/2004.12276
[code_amrcnn]: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/fashionpedia
