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
<br></br>

4. Generate Ground-Truths (masks and bboxes) using *GroundingDINO+SAM*: `/dino_sam/GroundingDINO+SAM.ipynb`
    - the raw annotations (*3,918*) are saved at `/data/fashionfail/annotations/`, e.g. `/adi_15_3.pkl` --> 1 pickle file per image, with *bbox, confidence(bbox), mask*
<br></br>

5. Quality check of GTs using `/fashionfail/label_gt.py`
    - ***1,105*** images (of 1,057 are fine) are labeled and results stored at `data/fashionfail/labeled_images_GT.json`
