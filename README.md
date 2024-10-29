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

#### Example table output

Sorted by self time, limited to 20 entries:

```
name                   class                       total_mean_time[ms]    child_mean_time[ms]    self_mean_time[ms]    times_run
---------------------  ------------------------  ---------------------  ---------------------  --------------------  -----------
BoundBoxPostProcessor  BoundBoxPostProcessor                 122.402                  0                   122.402              2
rpn                    RegionProposalNetwork                  75.1099                13.2938               61.8161            40
conv1                  Conv2d                                 56.7877                 0                    56.7877            40
transform              GeneralizedRCNNTransform               22.545                  0                    22.545             40
conv1                  Conv2d                                 15.7004                 0                    15.7004            40
Resize                 Resize                                 12.9608                 0                    12.9608             2
box_roi_pool           MultiScaleRoIAlign                     12.0589                 0                    12.0589            40
bn1                    FrozenBatchNorm2d                       9.38578                0                     9.38578           40
fc6                    Linear                                  8.57618                0                     8.57618           40
roi_heads              RoIHeads                               30.5489                22.8484                7.70046           40
anchor_generator       AnchorGenerator                         7.19101                0                     7.19101           40
head                   RPNHead                                 6.10277                1.18993               4.91284           40
conv2                  Conv2d                                  4.90413                0                     4.90413           40
conv2                  Conv2d                                  3.71823                0                     3.71823           40
relu                   ReLU                                    3.55662                0                     3.55662           40
0                      Conv2d                                  2.08268                0                     2.08268           40
conv2                  Conv2d                                  1.98001                0                     1.98001           40
cls_score              Linear                                  1.87586                0                     1.87586           40
conv3                  Conv2d                                  1.49347                0                     1.49347           40
conv1                  Conv2d                                  1.23685                0                     1.23685           40
```

#### Example tree output

Shows the entire pipeline:

```
Pipeline (PIPELINE) took 396.44ms (self: 0.00ms)
. Preprocessing (PREPROCESSING) took 12.96ms (self: 0.00ms)
. . Resize (Resize) took 12.96ms (self: 12.96ms)
. . ToTensor (ToTensor) took 0.00ms (self: 0.00ms)
. Model (FasterRCNN) took 261.08ms (self: 0.75ms)
. . transform (GeneralizedRCNNTransform) took 22.55ms (self: 22.55ms)
. . backbone (BackboneWithFPN) took 132.13ms (self: 0.02ms)
. . . body (IntermediateLayerGetter) took 127.69ms (self: 0.08ms)
. . . . conv1 (Conv2d) took 56.79ms (self: 56.79ms)
. . . . bn1 (FrozenBatchNorm2d) took 9.39ms (self: 9.39ms)
. . . . relu (ReLU) took 3.56ms (self: 3.56ms)
. . . . maxpool (MaxPool2d) took 0.60ms (self: 0.60ms)
. . . . layer1 (Sequential) took 23.68ms (self: 0.03ms)
. . . . . 0 (Bottleneck) took 20.51ms (self: 0.21ms)
. . . . . . conv1 (Conv2d) took 15.70ms (self: 15.70ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.24ms (self: 0.24ms)
. . . . . . conv2 (Conv2d) took 1.98ms (self: 1.98ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . conv3 (Conv2d) took 1.49ms (self: 1.49ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . relu (ReLU) took 0.03ms (self: 0.03ms)
. . . . . . downsample (Sequential) took 0.36ms (self: 0.02ms)
. . . . . . . 0 (Conv2d) took 0.08ms (self: 0.08ms)
. . . . . . . 1 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . 1 (Bottleneck) took 1.93ms (self: 0.18ms)
. . . . . . conv1 (Conv2d) took 0.84ms (self: 0.84ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv2 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.22ms (self: 0.22ms)
. . . . . . conv3 (Conv2d) took 0.08ms (self: 0.08ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . . relu (ReLU) took 0.03ms (self: 0.03ms)
. . . . . 2 (Bottleneck) took 1.21ms (self: 0.19ms)
. . . . . . conv1 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . conv2 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.24ms (self: 0.24ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.22ms (self: 0.22ms)
. . . . . . relu (ReLU) took 0.03ms (self: 0.03ms)
. . . . layer2 (Sequential) took 10.65ms (self: 0.03ms)
. . . . . 0 (Bottleneck) took 5.99ms (self: 0.24ms)
. . . . . . conv1 (Conv2d) took 0.48ms (self: 0.48ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv2 (Conv2d) took 3.72ms (self: 3.72ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv3 (Conv2d) took 0.19ms (self: 0.19ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . . downsample (Sequential) took 0.60ms (self: 0.02ms)
. . . . . . . 0 (Conv2d) took 0.35ms (self: 0.35ms)
. . . . . . . 1 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . 1 (Bottleneck) took 1.88ms (self: 0.22ms)
. . . . . . conv1 (Conv2d) took 0.46ms (self: 0.46ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . . conv2 (Conv2d) took 0.26ms (self: 0.26ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.27ms (self: 0.27ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.27ms (self: 0.27ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . 2 (Bottleneck) took 1.43ms (self: 0.23ms)
. . . . . . conv1 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv2 (Conv2d) took 0.13ms (self: 0.13ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.27ms (self: 0.27ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.30ms (self: 0.30ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . 3 (Bottleneck) took 1.33ms (self: 0.21ms)
. . . . . . conv1 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv2 (Conv2d) took 0.12ms (self: 0.12ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.27ms (self: 0.27ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . layer3 (Sequential) took 11.17ms (self: 0.04ms)
. . . . . 0 (Bottleneck) took 2.67ms (self: 0.22ms)
. . . . . . conv1 (Conv2d) took 0.23ms (self: 0.23ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.24ms (self: 0.24ms)
. . . . . . conv2 (Conv2d) took 0.77ms (self: 0.77ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.20ms (self: 0.20ms)
. . . . . . conv3 (Conv2d) took 0.24ms (self: 0.24ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . . downsample (Sequential) took 0.48ms (self: 0.02ms)
. . . . . . . 0 (Conv2d) took 0.22ms (self: 0.22ms)
. . . . . . . 1 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . 1 (Bottleneck) took 2.49ms (self: 0.20ms)
. . . . . . conv1 (Conv2d) took 1.24ms (self: 1.24ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv2 (Conv2d) took 0.21ms (self: 0.21ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.23ms (self: 0.23ms)
. . . . . . conv3 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . 2 (Bottleneck) took 1.34ms (self: 0.21ms)
. . . . . . conv1 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.25ms (self: 0.25ms)
. . . . . . conv2 (Conv2d) took 0.12ms (self: 0.12ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . relu (ReLU) took 0.04ms (self: 0.04ms)
. . . . . 3 (Bottleneck) took 1.50ms (self: 0.23ms)
. . . . . . conv1 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv2 (Conv2d) took 0.14ms (self: 0.14ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv3 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.34ms (self: 0.34ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . . . 4 (Bottleneck) took 1.46ms (self: 0.23ms)
. . . . . . conv1 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv2 (Conv2d) took 0.14ms (self: 0.14ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.27ms (self: 0.27ms)
. . . . . . conv3 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.31ms (self: 0.31ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . . . 5 (Bottleneck) took 1.66ms (self: 0.23ms)
. . . . . . conv1 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv2 (Conv2d) took 0.14ms (self: 0.14ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.28ms (self: 0.28ms)
. . . . . . conv3 (Conv2d) took 0.09ms (self: 0.09ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.50ms (self: 0.50ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . . layer4 (Sequential) took 11.78ms (self: 0.02ms)
. . . . . 0 (Bottleneck) took 7.60ms (self: 0.26ms)
. . . . . . conv1 (Conv2d) took 0.21ms (self: 0.21ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.30ms (self: 0.30ms)
. . . . . . conv2 (Conv2d) took 4.90ms (self: 4.90ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.26ms (self: 0.26ms)
. . . . . . conv3 (Conv2d) took 0.82ms (self: 0.82ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.29ms (self: 0.29ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . . . . downsample (Sequential) took 0.51ms (self: 0.02ms)
. . . . . . . 0 (Conv2d) took 0.19ms (self: 0.19ms)
. . . . . . . 1 (FrozenBatchNorm2d) took 0.30ms (self: 0.30ms)
. . . . . 1 (Bottleneck) took 2.55ms (self: 0.24ms)
. . . . . . conv1 (Conv2d) took 0.99ms (self: 0.99ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.30ms (self: 0.30ms)
. . . . . . conv2 (Conv2d) took 0.25ms (self: 0.25ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.29ms (self: 0.29ms)
. . . . . . conv3 (Conv2d) took 0.11ms (self: 0.11ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.31ms (self: 0.31ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . . . 2 (Bottleneck) took 1.61ms (self: 0.25ms)
. . . . . . conv1 (Conv2d) took 0.10ms (self: 0.10ms)
. . . . . . bn1 (FrozenBatchNorm2d) took 0.32ms (self: 0.32ms)
. . . . . . conv2 (Conv2d) took 0.15ms (self: 0.15ms)
. . . . . . bn2 (FrozenBatchNorm2d) took 0.30ms (self: 0.30ms)
. . . . . . conv3 (Conv2d) took 0.12ms (self: 0.12ms)
. . . . . . bn3 (FrozenBatchNorm2d) took 0.32ms (self: 0.32ms)
. . . . . . relu (ReLU) took 0.05ms (self: 0.05ms)
. . . fpn (FeaturePyramidNetwork) took 4.42ms (self: 0.80ms)
. . . . inner_blocks (ModuleList) took 0.72ms (self: 0.00ms)
. . . . . 0 (Conv2dNormActivation) took 0.23ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.22ms (self: 0.22ms)
. . . . . 1 (Conv2dNormActivation) took 0.13ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.12ms (self: 0.12ms)
. . . . . 2 (Conv2dNormActivation) took 0.13ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.12ms (self: 0.12ms)
. . . . . 3 (Conv2dNormActivation) took 0.23ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.22ms (self: 0.22ms)
. . . . layer_blocks (ModuleList) took 2.84ms (self: 0.00ms)
. . . . . 0 (Conv2dNormActivation) took 0.28ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.27ms (self: 0.27ms)
. . . . . 1 (Conv2dNormActivation) took 0.28ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.27ms (self: 0.27ms)
. . . . . 2 (Conv2dNormActivation) took 0.18ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.17ms (self: 0.17ms)
. . . . . 3 (Conv2dNormActivation) took 2.09ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 2.08ms (self: 2.08ms)
. . . . extra_blocks (LastLevelMaxPool) took 0.06ms (self: 0.06ms)
. . rpn (RegionProposalNetwork) took 75.11ms (self: 61.82ms)
. . . anchor_generator (AnchorGenerator) took 7.19ms (self: 7.19ms)
. . . head (RPNHead) took 6.10ms (self: 4.91ms)
. . . . conv (Sequential) took 0.39ms (self: 0.01ms)
. . . . . 0 (Conv2dNormActivation) took 0.38ms (self: 0.01ms)
. . . . . . 0 (Conv2d) took 0.33ms (self: 0.33ms)
. . . . . . 1 (ReLU) took 0.04ms (self: 0.04ms)
. . . . cls_logits (Conv2d) took 0.63ms (self: 0.63ms)
. . . . bbox_pred (Conv2d) took 0.17ms (self: 0.17ms)
. . roi_heads (RoIHeads) took 30.55ms (self: 7.70ms)
. . . box_roi_pool (MultiScaleRoIAlign) took 12.06ms (self: 12.06ms)
. . . box_head (TwoMLPHead) took 8.81ms (self: 0.14ms)
. . . . fc6 (Linear) took 8.58ms (self: 8.58ms)
. . . . fc7 (Linear) took 0.09ms (self: 0.09ms)
. . . box_predictor (FastRCNNPredictor) took 1.98ms (self: 0.02ms)
. . . . cls_score (Linear) took 1.88ms (self: 1.88ms)
. . . . bbox_pred (Linear) took 0.08ms (self: 0.08ms)
. Postprocessing (POSTPROCESSING) took 122.40ms (self: 0.00ms)
. . BoundBoxPostProcessor (BoundBoxPostProcessor) took 122.40ms (self: 122.40ms)
```
