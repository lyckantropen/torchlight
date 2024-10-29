import torch
from PIL import Image, ImageDraw
from pathlib import Path

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


class BoundBoxPostProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, paths, predictions):
        out_images = []
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
            out_images.append((out_path, draw_image))
        return out_images
