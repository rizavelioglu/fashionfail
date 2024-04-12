# Post-Train

#### Experiment Tracking
Run the following within the repository to start the `aim` server:
```bash
aim up
```
which returns the server URL, *e.g.* `http://127.0.0.1:43800`.
Head over to that URL or `http://localhost:43800` in your browser to see the `aim` UI.



#### Convert a model to `ONNX` format
Execute the following command inside the repo (which will save `.onnx` model to `./saved_models`):
```bash
python segmentation/models/export_to_onnx.py \
--ckpt_path "model.ckpt" \
--onnx_path "model.onnx"
```


---
<div style="display: flex; justify-content: space-between;">

   [Back](03_training.md)

   [Next: Evaluation](05_evaluation.md)

</div>
