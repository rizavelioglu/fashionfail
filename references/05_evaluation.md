# Evaluation
Models output predictions in different formats (bboxes and categories), as shown in [Inference docs][docs_inference].
Therefore, we unify all the predictions, i.e. convert bboxes to COCO format and map the category labels accordingly.

First, make sure we are in the correct environment and directory:
```bash
conda activate ff
cd /change/dir/to/fashionfail/repo/
```

Then, run evaluation with the following script:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "path/to/predictions/file" \         # either ends with `.npy` or `.npz`
    --anns_path "./cache/fashionfail/ff_test.json" \  # Annotation file of the dataset (Ground-Truth)
    --eval_method "COCO" \   # either one of [`COCO`, `COCO-extended`, `all`]
    --model_name "amrcnn" \  # either one of [`amrcnn`, `fformer`, `facere_plus`]
    --iou_type "bbox"        # either one of [`bbox`, `segm`]
```
which initially converts the raw predictions (`preds_path`) to COCO format and exports a `.json` file to the same
directory. Then, it executes the original COCO evaluation (`eval_method="COCO"`) with the specified `iou_type="bbox"`
on the specified dataset `anns_path` for the specified model: `model_name="amrcnn"`.

The script first prints the (official) COCO evaluation results, then prints $AP$ per class. Finally, the
inverse-frequency weighted AP, namely $mAP_w$ (the main evaluation metric in the paper) is also printed out to the
console. In addition, the TP, FP, and FN counts per category are also printed when `eval_method=COCO-extended` passed.

> Note on the `eval_method=COCO-extended`: \
> The official COCO evaluation does not return the true positive (TP), false positive (FP), and false negative (FN)
> counts, and the community has not provided a solution for this. Therefore, we have added that functionality to the
> official COCOEval script with minimal additions and named is `COCOEval2`. As we use the `black` code formatter, the
> diff between the original and custom implementation would include unnecessary changes. Hence, one could search for
> `COCOEval2` inside our custom implementation to see the changes, [click here to see changes][cocoeval2_changes].




# Evaluation Results

We leveraged the evaluation metrics used by [COCO][coco_eval] and present the results on 2 datasets;
`FashionFail-test` and `Fashionpedia-val`. The metrics are calculated for both boxes and segmentation masks and
shown in `box | mask` format.


### `FashionFail-test`:
Below are the evaluation scripts to reproduce the results presented in the paper.

| Model             |       $mAP_w$        |  $mAP_w^{@IoU=.50}$  |  $mAP_w^{@IoU=.75}$  |    $mAR_w^{top1}$    |   $mAR_w^{top100}$   |
|:------------------|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| `amrcnn-spine`    |     21.0 \| 21.2     |     22.1 \| 22.0     |     21.5 \| 21.5     |     32.1 \| 32.1     |     32.7 \| 32.7     |
| `fformer-swin`    |     22.2 \| 20.9     |     25.1 \| 23.2     |     21.5 \| 20.2     |     32.9 \| 30.2     |     44.6 \| 40.0     |
| `amrcnn-r50-fpn`  |     18.3 \| 18.6     |     19.7 \| 19.2     |     19.3 \| 19.0     |     25.7 \| 25.6     |     25.8 \| 25.6     |
| `fformer-r50-fpn` |     14.4 \| 14.5     |     14.6 \| 14.6     |     14.6 \| 14.6     |     25.9 \| 25.9     |     26.9 \| 26.3     |
| `facere`          | **24.0** \| **24.3** | **27.5** \| **26.8** | **25.1** \| **24.9** | **44.6** \| **45.5** | **47.5** \| **47.9** |
| `facere+`         |     93.4 \| 94.1     |     95.7 \| 95.7     |     95.3 \| 95.6     |     96.6 \| 97.3     |     96.6 \| 97.3     |

<details>
  <summary>Scripts</summary>

`amrcnn-spine`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "tpu/models/official/detection/outputs/spinenet143-ff_test.npy" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "amrcnn" --iou_type "bbox"
```

`fformer-swin`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "FashionFormer/outputs/fashionformer_swin_b_3x-ff-test.npz" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "fformer" --iou_type "bbox"
```

`amrcnn-r50-fpn`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "tpu/models/official/detection/outputs/r50fpn-ff_test.npy" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "amrcnn" --iou_type "bbox"
```

`fformer-r50-fpn`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "FashionFormer/outputs/fashionformer_swin_b_3x-ff-test.npz" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "fformer" --iou_type "bbox"
```

`facere`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "outputs/facere-ff_test.npz" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "facere" --iou_type "bbox"
```

`facere+`:
```bash
python src/fashionfail/models/evaluate.py \
    --preds_path "outputs/facere_plus-ff_test.npz" \
    --anns_path "./cache/fashionfail/ff_test.json" \
    --eval_method "COCO-extended" --model_name "facere_plus" --iou_type "bbox"
```

</details>




---

### `Fashionpedia-val`:
Since the `Fashionpedia-test` annotations are not publicly available, we evaluated all the models on the
`Fashionpedia-val` dataset, to make a fair comparison.


| Model             |       $mAP_w$        |  $mAP_w^{@IoU=.50}$  |  $mAP_w^{@IoU=.75}$  |    $mAR_w^{top1}$    |   $mAR_w^{top100}$   |
|:------------------|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| `amrcnn-spine`    |     66.6 \| 60.1     |     84.3 \| 82.0     |     73.7 \| 67.1     |     65.2 \| 60.9     |     75.5 \| 69.0     |
| `fformer-swin`    | **72.4** \| **70.6** | **87.6** \| **90.0** | **77.9** \| **77.3** | **71.1** \| **69.1** | **81.9** \| **78.4** |
| `amrcnn-r50-fpn`  |    64.3  \| 58.3     |  83.2      \|  80.6  |  72.7      \| 64.9   | 63.4         \| 60.0 |  73.4       \| 67.6  |
| `fformer-r50-fpn` |  64.4    \|   64.2   |  82.4   \|    85.0   |   69.3  \|    69.9   |  65.8   \|     65.0  |  77.5   \|    74.6   |
| `facere`          | 63.9     \|     58.0 |   82.5 \|     80.3   |   71.6 \|     64.3   |   64.5 \|     60.8   |   76.0 \|    69.4    |



<span style="color:red">add scripts as above (inside 00_paper.ipynb)</span>
<details>
  <summary>Scripts</summary>

`amrcnn-spine`:

</details>


[coco_eval]: https://cocodataset.org/#detection-eval
[docs_inference]: 04_inference.md
[cocoeval2_changes]: https://github.com/search?q=COCOeval2+repo%3Arizavelioglu%2Ffashionfail+path%3A**%2Fcocoeval2.py&type=code&ref=advsearch

---
<div style="display: flex; justify-content: space-between;">

   [Back](04_inference.md)

   [Next: Visualization](06_visualization.md)

</div>
