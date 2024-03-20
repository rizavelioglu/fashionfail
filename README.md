# FashionFail

### Models
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


### Project Structure
The following project/directory structure is adopted:
[Cookiecutter Data Science by DrivenData](https://drivendata.github.io/cookiecutter-data-science/).


### FAQ
**[Q]** _Why didn't we scrape the category information and avoid using a LLM?_
>The scraped categories do not align with the Fashionpedia ontology. As a result, manual category mapping is
necessary, which is time-consuming and prone to errors.

**[Q]** _Why only a subset of categories are used?_
>To make it possible to automatize the annotation pipeline, which is designed to provide only a single
annotation for an image. Providing multiple annotations per image automatically is extremely challenging. For example,
consider a shoe image with an applique on it which has 2 annotations; _"applique"_ and _"shoe"_. Multiple issues may occur:
>- how to annotate label: "applique" can not be extracted from product descriptions,
>- GroundingDINO's performance differs for different labels: boxes for "applique" may exhibit less accuracy than those for "shoe",
>- mask ambiguity: SAM provides 3 masks for each box (to solve 'ambiguity') and we select the one with the highest score.
However, for small boxes (e.g. applique) the correct mask may not be the one with the highest score, hence, manual
labeling is required.

**[Q]** _Can SAM be used for fashion object segmentation, which would eliminate training a specific model for this task?_
>SAM is great at segmenting anything however, it does not classify the segmented masks. In addition, SAM
produces many masks (segmenting literally anything), for example, for a shoe image it may provide different masks for the shoelace,
applique, shoe sole, etc. which requires a further method to combine those masks into a single one.
