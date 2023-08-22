# FashionFail

Here's the full pipeline of creating FashionFail Dataset:
1. Scrape the following websites; `adidas`, `newbalance`, `puma`, `birkenstock`.
    - scraped contents are saved in `data/fashionfail/` in respective folders, e.g. `./adidas/raw/0.json` (1 json file per item)
    - all the json files are preprocessed and merged in `/fashionfail/notebooks/01_process_raw_data.ipynb`.\
        - The merged data is saved to `/work/data/fashionfail/df_all.csv`\
        - Images (***116k***) are downloaded using that merged .csv file to `/data/fashionfail/images/`
<br></br>

2. Labeled images as *"single/clean"* images using `/fashionfail/label_single_objects.py`
    - ***10,860*** images are labeled and results are stored at `/data/fashionfail/labeled_images.json`
<br></br>

3. Label categories of the images using *text-davinci-003* in `/fashionfail/notebooks/02_label_cat_gpt3-5.ipynb`
    - make preds for only the *single* images labeled previously.
    - save results in `/data/fashionfail/preds_categories.json`
    - process raw results and save processed data to `/data/fashionfail/df_cat.csv`
    - merge `df_cat` with `df_all` and save to `/data/fashionfail/df_cat-processed.csv`
    - ***3,897*** images
<br></br>

4. Generate Ground-Truths (masks and bboxes) using *GroundingDINO+SAM*: `/dino_sam/GroundingDINO+SAM.ipynb`
    - the raw annotations (*3,901*) are saved at `/data/fashionfail/annotations/`, e.g. `/adi_15_3.pkl` --> 1 pickle file per image, with *bbox, confidence(bbox), mask*
<br></br>

5. Quality check of GTs using `/fashionfail/label_gt.py`
    - ***1,096*** images (of 1,015 are fine) are labeled and results stored at `/data/fashionfail/labeled_images_GT.json`
<br></br>

<details>
  <summary>[outdated] 6. "Complete" generated GTs.\</summary>
Generated GTs are not complete. For example, for a t-shirt a box is generated automatically for the class top, t-shirt
but other detections are missing such as; 'sleeves', 'neckline', etc. Therefore, such GTs need to be added. Since we
don't want to annotate images manually, we used `AMRCNN` to make predictions which make pretty well predictions for
those classes. After that, we manually check the predicted bounding boxes and choose the ones that are correct.
   - script: `/fashionfail/label_gt_boxes.py`,
   - results: `/data/fashionfail/labeled_images_GT_boxes.json`
   - ***302*** images (out of **1,057** images from step 5.) are labeled.
   - <span style="color:red">**TODO**</span>: add these labeled GTs to the auto-generated GTs and finalize the dataset.
</details>

6. Move labeled images to a new folder:
```python
with open("/home/rizavelioglu/work/data/fashionfail/labeled_images_GT.json", "r+") as f:
    labeled_images_data = json.load(f)
    
import shutil
src = "/home/rizavelioglu/work/data/fashionfail/images/"
dst = "/home/rizavelioglu/work/data/fashionfail/images-sample/"

for im in labeled_images_data["images_to_keep"]:
    shutil.copyfile(src+im, dst+im)
```

7. Tar `images-sample` and get predictions from `tpu`
