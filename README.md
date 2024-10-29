# Tools for timing object detection pipelines

## How to run

Python 3.10+ is required.

To create the environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To run the example script (Faster R-CNN model with a ResNet-50-FPN backbone):

```bash
python evaluate_rcnn.py
```

The script will run on the CPU first and on a CUDA device if it is available.

The script will use the files `test.jpg` and `test_2.jpg` and create files
`test_targets.jpg` and `test_2.jpg` with bounding boxes.

## Explanation

The package `ktroj_torchlight` provides a context manager called `ModelTiming`
that accepts the model object and two lists of transforms: the preprocessing
transforms and the postprocessing transforms.

The context manager adds instrumentation to the module and transforms that
replaces the `forward` method (or `__call__` if the former is not present) that
times the actual execution of the module at hand. It does not execute any of the
transforms or do any inference on the model itself, this is assumed to happen
within the active context.

After the context manager is released, the timing data is accumulated and can be
summarized by using methods `summarize_table` and `summarize_tree`.

### Example

```python
model = model.to(device)

post_transform = BoundBoxPostProcessor()

with ModelTiming(model, pre_transforms, [post_transform]):
    for images, paths in dataloader:
        images = images.to(device)
        predictions = model(images)

        # required for the timing to be accurate
        torch.cpu.synchronize()
        if device == 'cuda':
            torch.cuda.synchronize()

        outputs = post_transform(images, paths, predictions)

print(f'Pipeline time: {timing.timing_data.total_time*1000:.2f}ms')
print(timing.summarize_table())
```
