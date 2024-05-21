# Documentation

Here we detail every step performed inside the repository.

- [01_dataset_creation](01_dataset_creation.md):
  - scraping, filtering, annotation, quality review, and constructing COCO-style dataset.
- [02_data-analysis](02_data_analysis.md):
  - brief exploratory data analysis.
- [03_training](03_training.md):
  - `Facere`, `Facere+` model architectures, training, experiment tracking.
- [04_inference](04_inference.md):
  - Converting trained models to `ONNX` format, inference on both `Facere` and `Facere+` and other SoTA models such as;
  `Attribute Mask R-CNN` and `Fashionformer`.
- [05_evaluation](05_evaluation.md):
  - Evaluation metrics and results.
- [06_visualization](06_visualization.md):
  - Visualizing model predictions (bounding boxes and segmentation masks), analysis on False Positives, various
  visualizations including Reliability Diagrams.
- [FAQ](faq.md):
  - Other details regarding pipeline, dataset, further work, and closing remarks.

---
<div align="right">

   [Next: Dataset Creation](01_dataset_creation.md)

</div>
