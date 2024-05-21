# Training

We proposed 2 trained models in the paper; `Facere` and `Facere+`. In the following we explain the steps required for
training.


### Experiment Tracking
We use the open-source [aim][aim_website] for experiment tracking. Before training, initialize aim repository where all
training runs are logged:
```bash
aim init   # this is done only once
```

**_After training_**, run the following within the repository to start the `aim` server:
```bash
aim up
```
which returns the server URL, *e.g.* `http://127.0.0.1:43800`.
Head over to that URL or `http://localhost:43800` in your browser to see the `aim` UI.


### _Facere_
The Fashionpedia dataset comprises three splits: train, validation,
and test. However, annotations for the _**test set**_ are not publicly available for genuine testing purposes. As a
result, we divide the `Fashionfail-train` dataset into two subsets: one for training and another for validation.
Subsequently, we utilize the `Fashionpedia-val` dataset as a surrogate "test" set, which serves as a basis for comparing
model performances, as we will see in the next section.

The following script prepares the dataset for training, which includes;
- downloading Fashionpedia dataset from the [official source](https://github.com/cvdfoundation/fashionpedia),
- splitting training data into train and val sets with a split rate of 75%-25%, respectively,
- converting the mask annotations to encoded RLE format.

```bash
python fashionfail/data/make_fashionpedia.py
```

To initiate training on the split train set and validate it on the split validation set, execute the following command:
```bash
python fashionfail/models/train_facere_base.py  \
--train_img_dir "~/.cache/fashionpedia/images/train/" \
--val_img_dir "~/.cache/fashionpedia/images/train/" \    # same as train_img_dir
--train_ann "~/.cache/fashionpedia/fp_train_rle.json" \
--val_ann "~/.cache/fashionpedia/fp_val_rle.json" \
--batch_size 8 \
--shuffle True \
--accelerator gpu \
--devices 1 \
--precision 16 \
--max_epochs 50 \
--strategy "ddp" \
--optimizer "adam" \
--experiment-name "facere_base"
```

> **Note:** When using the split train and validation sets, it's important to note that the images for both splits are
> located within `.../images/train/` directory, as per the official organization of the `Fashionpedia-train` dataset.

Also, see [PytorchLightning's docs][pl_docs] for all the arguments available for`pl.Trainer()`.



### _Facere+_
To show that `FashionFail Dataset` is "learnable" and noise-free, we take the `Facere` model (trained as above) and
fine-tune it on `FashionFail-train` while validating on `FashionFail-val`:

```bash
python fashionfail/models/train_facere_plus.py \
--train_img_dir "~/.cache/fashionfail/images/train/" \
--val_img_dir "~/.cache/fashionfail/images/val/" \
--train_ann "~/.cache/fashionfail/ff_train.json" \
--val_ann "~/.cache/fashionfail/ff_val.json" \
--batch_size 8 \
--shuffle True \
--accelerator "gpu" \
--devices 1 \
--precision 16 \
--max_epochs 150 \
--strategy "ddp" \
--optimizer "adam" \
--experiment-name "facere_plus" \
--pretrained_model_path "./saved_models/facere_base.ckpt"
```

---
<div style="display: flex; justify-content: space-between;">

   [Back](02_data_analysis)

   [Next: Inference](04_inference.md)

</div>

[pl_docs]: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html
[aim_website]: https://github.com/aimhubio/aim
