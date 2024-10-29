import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ktroj_torchlight import (BoundBoxPostProcessor, FileListDataset,
                              ModelTiming)

image_paths = [
    "test.jpg",
    "test_2.jpg"
]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
post_transforms = [BoundBoxPostProcessor()]

dataset = FileListDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

model = fasterrcnn_resnet50_fpn(pretrained=True)

for device in devices:
    print(f'Running on {device}')
    model = model.to(device)
    model.eval()

    with ModelTiming(model, pre_transforms=transform.transforms, post_transforms=post_transforms) as timing:
        for images, paths in dataloader:
            images = images.to(device)
            for _ in range(10):
                predictions = model(images)

                torch.cpu.synchronize()
                if device == 'cuda':
                    torch.cuda.synchronize()

            for post_transform in post_transforms:
                out_images = post_transform(images, paths, predictions)
                print(out_images)

    print(f'Pipeline time: {timing.timing_data.total_time * 1000:.2f}ms')
    print(timing.summarize_table(limit=20))
    print(timing.summarize_tree())
