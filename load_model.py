import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
import numpy as np


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
        return image


image_paths = [
    "test.jpg",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a tensor
])

dataset = CustomDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()


class ForwardTimer:
    def __init__(self, module, times):
        self.module = module
        self._true_forward = module.forward
        self._times = times

    def __call__(self, *args, **kwargs):
        # if isinstance(self.module, torch.nn.ModuleList) or isinstance(self.module, torch.nn.ModuleDict):
        #     return self._true_forward(*args, **kwargs)

        start = time.perf_counter()
        result = self._true_forward(*args, **kwargs)
        end = time.perf_counter()
        self._times.append(end - start)
        # print(f"{self.module.__class__.__name__} took {end - start} seconds")
        return result

    def disable(self):
        self.module.forward = self._true_forward

    # def __del__(self):
    #     self.module.forward = self._true_forward


def add_instrumentation(module, structure, forward_timers, name="", level=0):
    for name, child in module.named_children():
        # print(''.join(["\t"]*level) + name)
        child_structure = {'name': name, 'class': child.__class__.__name__, 'times': [],
                           'children': []}
        add_instrumentation(child, child_structure,
                            forward_timers, name, level+1)
        child.forward = ForwardTimer(child, child_structure['times'])
        forward_timers.append(child.forward)
        # child_structure['forward'] = child.forward
        structure['children'].append(child_structure)


forward_timers = []
structure = {'name': "fasterrcnn_resnet50_fpn", 'times': [],
             'children': [], 'class': model.__class__.__name__}
add_instrumentation(model, structure, forward_timers)

for image in dataloader:
    image = image.to(device)
    for _ in range(100):
        begin = time.perf_counter()
        predictions = model(image)
        torch.cpu.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        structure['times'].append(end - begin)
    print(predictions)

for forward_timer in forward_timers:
    forward_timer.disable()

del forward_timers


def summarize_times(s):
    child_times = [summarize_times(child) for child in s['children']]
    s['cumulative_time'] = sum(child_times) if len(child_times) > 0 else 0.0
    if len(s['times']) > 0:
        s['self_time'] = np.mean(s['times']) - s['cumulative_time']
        s['total_time'] = np.mean(s['times'])
    else:
        s['self_time'] = 0
        s['total_time'] = s['cumulative_time']
    return s['total_time']


summarize_times(structure)


def print_structure(s, level=0):
    if len(s['times']) > 0:
        print(''.join([". "]*level) + s['name'] + f' ({s["class"]})' +
              f" took {np.mean(s['times']) * 1000:.2f} ms ({s['self_time'] * 1000:.2f} ms)")
    else:
        print(''.join([". "]*level) + s['name'] +
              f' ({s["class"]}) ({s["cumulative_time"] * 1000:.2f} ms)')
    for child in s['children']:
        print_structure(child, level+1)


print_structure(structure)

# def draw_boxes(image, boxes):
#     draw = ImageDraw.Draw(image)
#     for box in boxes:
#         xmin, ymin, xmax, ymax = box
#         draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
#     return image

# for images in dataloader:
#     predictions = model(images)
#     for i, image in enumerate(images):
#         boxes = predictions[i]['boxes'].detach().cpu().numpy()
#         image_with_boxes = draw_boxes(image, boxes)
#         image_with_boxes.show()
