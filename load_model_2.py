from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ktroj_torchlight import ModelTiming, ModelTimingInner


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


image_paths = [
    "test.jpg",
    "test_2.jpg"
]

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a tensor
])

dataset = CustomDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

with ModelTiming(model) as timing:
    for images, paths in dataloader:
        images = images.to(device)
        for _ in range(10):
            with ModelTimingInner(timing):
                predictions = model(images)

            torch.cpu.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for image, path, prediction in zip(images, paths, predictions):
            # Draw bounding boxes on the image
            draw_image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(draw_image)
            scale_y, scale_x = draw_image.size[1] / image.shape[-2], draw_image.size[0] / image.shape[-1]

            colors = ["red", "green", "blue", "yellow", "purple", "orange"]
            boxes = prediction['boxes'].detach().cpu().numpy()
            for box, score, label in zip(boxes, prediction['scores'], prediction['labels']):
                if score < 0.5:
                    continue
                color = colors[label % len(colors)]
                box = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
                draw.rectangle(box, outline=color, width=2)
                coord = (box[0], box[1])
                draw.text(coord, f"{COCO_INSTANCE_CATEGORY_NAMES[label]}", fill=color, font_size=20)
                coord = (box[0], box[1] + 20)
                draw.text(coord, f"{score:.2f}", fill=color, font_size=15)

            out_path = Path(path).stem + "_targets.jpg"
            draw_image.save(out_path)

print(f'Model time: {timing.timing_data.total_time}')
print(timing.summarize_table())
print(timing.summarize_tree())
