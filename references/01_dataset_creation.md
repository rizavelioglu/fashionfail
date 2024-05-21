# Dataset Creation

### 1. Scraping

We developed a web crawler specifically tailored for adidas.de using [Scrapy][website_scrapy].
However, due to concerns about data privacy regulations, we've opted not to share it publicly.

We scraped the adidas website and collected the following data:

```json
{
   "date": "2023",
   "url": "https://www.adidas.de/samba-og-schuh/B75807.html",
   "productname": "SAMBA OG SCHUH",
   "description": "Vom Fußballschuh zum Streetwear-Favourite. Mit seiner cleanen Low-Top-Silhouette, dem weichen Leder-Obermaterial mit Wildleder-Overlays und der Außensohle aus Naturgummi ist der Samba ein echter Evergreen und passt einfach immer und überall.",
   "images": [
      "https://assets.adidas.com/images/4c70105150234ac4b948a8bf01187e0c_9366/Samba_OG_Schuh_Schwarz_B75807_01_standard.jpg",
      "https://assets.adidas.com/images/309a0c8f53dd45d3a3bea8bf0118aa6b_9366/Samba_OG_Schuh_Schwarz_B75807_02_standard_hover.jpg",
      ...
   ]
}
```


### 2. (Manual) Filtering
After downloading all the images using the scraped URLs (`"images"` column shown above), a manual filtering is performed. This is required as e-commerce
websites usually have multiple images for each product featuring the product from various views, as well as a human
wearing it, e.g. [see the product shown above][adidas_shoe].

We built a simple annotation tool, based on `tkinter`(the standard Python interface to the Tk GUI toolkit), for labeling
images based on three strict criteria;
1. presence of multiple objects or instances of the same object,
2. visibility of any part of the human body,
3. extreme close-ups hindering category determination from the image.

Figure 3 in the paper illustrates examples for each criterion. Images meeting any criterion were excluded to ensure
pure, clean, and informative e-commerce product images devoid of contextual information.

The tool processes each scraped image sequentially, displaying them one at a time for human annotation. The human
annotator can label each image using either the graphical user interface (GUI) or keyboard shortcuts. Start the process
with:
```bash
python data/label_single_objects.py \
    --images_dir="/path/to/images/folder" \
    --out_path="~/.cache/fashionfail/labeled_images.json"
```

The labels are stored in a `.json` file at `out_path` with the following structure:
```json
{
    "images_to_keep":    ["img_name_1", "..."],
    "images_to_discard": ["img_name_2", "..."]
}
```

All images labeled as `"images_to_discard"` _must be_ removed from the `images_dir`. This is necessary because
the following scripts assume that all images inside the images directory adhere to the three criteria explained above.


### 3. (Automatic) Annotating
For all the remaining images after filtering, we annotated the _category_, _bounding box_, and _segmentation mask_
annotations automatically using various foundational models.

#### a. Category Annotation
We leveraged the GPT3.5 model (`text-davinci-003`) by OpenAI to annotate the category of the product given its
description (which was scraped). However, due to concerns about data privacy, we do not share the scraped descriptions.

Nonetheless, we offer a short code snippet demonstrating how we accomplished it:
Considering the above example, the LLM was prompted with:
```python
item_description = "Vom Fußballschuh zum Streetwear-Favourite. Mit seiner cleanen Low-Top-Silhouette, dem weichen Leder-Obermaterial mit Wildleder-Overlays und der Außensohle aus Naturgummi ist der Samba ein echter Evergreen und passt einfach immer und überall."
instructions = ("List of categories: \"shirt, blouse\", \"top, t-shirt, sweatshirt\", \"sweater\", \"cardigan\", "
                "\"jacket\", \"vest\", \"pants\", \"shorts\", \"skirt\", \"coat\", \"dress\", \"jumpsuit\", "
                "\"cape\", \"glasses\", \"hat\", \"headband, head covering, hair accessory\", \"tie\", \"glove\", "
                "\"watch\", \"belt\", \"leg warmer\", \"tights, stockings\", \"sock\", \"shoe\", \"bag, wallet\", "
                "\"scarf\", \"umbrella\", \"hood\", \"other\".\nGiven the list of categories above, "
                "which category does the product with the following description belong to?")
prompt = f"\n\nDescription:\n\n{item_description}\nCategory:"

# Make API request to OpenAI model
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=instructions+prompt,
  temperature=1,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```
where the model, _ideally_, returns the correct category, i.e. 'shoe' in this case.
However, due to inconsistencies in the model's output—sometimes returning variations like 'Shoe' or 'shoes'—we conducted
_extensive_ post-processing.

This step assumes that a `.csv` file is generated in the following format:

|     |   image_name    | class_id |       images       |
|:---:|:---------------:|:--------:|:------------------:|
|  0  | adi_10143_1.jpg |    23    | "URL/of/image.jpg" |
|  1  | adi_6216_3.jpg  |    18    | "URL/of/image.jpg" |
| ... |       ...       |   ...    |        ...         |

> **Note:** On January 4, 2024, OpenAI [deprecated](https://platform.openai.com/docs/deprecations) the `text-davinci-003`
> model. The recommended replacement is `gpt-3.5-turbo-instruct` which may perform differently compared to our results.

#### b. Bounding Box and Mask Annotations
The provided script initially executes inference on GroundingDINO to generate bounding box annotations for each image
within a specified folder.
Subsequently, utilizing the bounding box coordinates in conjunction with the respective images, it applies
SegmentAnything to produce mask annotations. The resulting annotations are stored in the `out_dir` folder, with each
image corresponding to one `.pkl` file. Each `.pkl` file is named the same as the corresponding image.

```bash
python data/annotate_boxes_and_masks.py \
    --images_dir "/path/to/images/folder" \
    --out_dir "~/.cache/fashionfail/annotations/"    # bbox and mask annotations
```


### 4. (Manual) Quality Review
With the following script, we visualize each image along with its generated annotations (_category_, _bbox_, and _mask_) to
verify their accuracy. A human annotator labels each sample with either;
-  _yes_: indicating that the annotations are accurate,
- _no_: indicating that at least one annotation is not accurate.

To start the review, execute the following:
```bash
python data/label_gt.py
    --images_dir="/path/to/images/folder" \
    --anns_dir="~/.cache/fashionfail/annotations/" \       # bbox and mask annotations (generated above in 3.b)
    --cat_anns="~/.cache/fashionfail/category_anns.csv" \  # category annotations (generated above in 3.a)
    --out_path="~/.cache/fashionfail/labeled_images_gt.json"
```

The labels are stored in a `.json` file at `out_path` with the following structure:
```json
{
    "images_to_keep":    ["img_name_1", "..."],
    "images_to_discard": ["img_name_2", "..."]
}
```

After review, the images labeled as `"images_to_discard"` are either discarded or their issues have been resolved.

If they are discarded, they _must be removed_ from the `category_anns.csv` file. This is necessary because
the following script includes every entry in the `.csv` file when constructing the dataset.



### 5. Construct the dataset in COCO format:

Given all the annotations (bbox, mask, category) and images, the following script first splits the dataset into three
disjoint sets and saves three files at `out_dir`: `ff_train.json`, `ff_test.json`, and `ff_val.json`.
```bash
python fashionfail/data/construct_dataset_in_coco.py \
    --images_dir="/path/to/images/folder" \
    --anns_dir="~/.cache/fashionfail/annotations/" \       # bbox and mask annotations (generated above in 3.b)
    --cat_anns="~/.cache/fashionfail/category_anns.csv" \  # category annotations (generated above in 3.a)
    --out_dir="~/.cache/fashionfail/"
```


---
<div style="display: flex; justify-content: space-between;">

   [Back](README.md)

   [Next: Data Analysis](02_data_analysis.md)

</div>

[website_scrapy]: https://scrapy.org/
[adidas_shoe]: https://www.adidas.de/samba-og-schuh/B75807.html
