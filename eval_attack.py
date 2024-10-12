from coco.coco_utils import get_coco_using_paths, get_coco_api_from_dataset
from coco.coco_eval import CocoEvaluator
from vision.torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pytorch_faster_rcnn import PyTorchFasterRCNN
from RainPGD.utils import add_rain
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import presets
import torch
import glob
import os
import argparse

parser = argparse.ArgumentParser(description="Evaluating Adversarial Attack")
parser.add_argument("--data_folder", type=str, default='adv_data/coco_random_patch_100')
parser.add_argument("--ann_file", type=str, default='data/annotations/instances_val2017.json')
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--image_set", type=str, default='val')
parser.add_argument("--adv", action='store_true', default=False, help='evaluate adversarial data')
parser.add_argument("--rain_adv", action='store_true', default=False, help='evaluate RainPGD adversarial data')
parser.add_argument("--rain", action='store_true', default=False, help='evaluate for rain robustness')
parser.add_argument("--rain_prob", type=float, default=0.7, help='rain probability')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_type = 'rain' if args.rain_adv else 'normal'
data_dir = f'{args.data_folder}/{data_type}/{args.image_set}/data'
images_dir = f'{args.data_folder}/{data_type}/{args.image_set}/images'

# setup model
model = fasterrcnn_resnet50_fpn(pretrained=True)
art_model = PyTorchFasterRCNN(
        model=model,
        detector=None,
        clip_values=(0, 1.0),
        channels_first=False,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type=DEVICE,
        adaptive=False,
        defense=False,
        bpda=False,
        shape_completion=False,
        adaptive_to_shape_completion=False,
        simple_shape_completion=False,
        bpda_shape_completion=False,
        union=False
)


def collate_fn(batch):
    return tuple(zip(*batch))

# load data
dataset = get_coco_using_paths(images_dir, args.ann_file, transforms=presets.DetectionPresetEval(), image_set='val')
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
)

# evaluate
art_model.model.eval()
coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(loader.dataset), iou_types=['bbox'])
data_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

for batch_idx, data_file in enumerate(tqdm(data_files)):
    data = torch.load(data_file)
    x = data['x']
    y = data['y']
    x_adv = data['x_adv']

    if args.rain:
        rain_type = np.random.choice(['weak', 'heavy', 'torrential'])
        if np.random.random() <= args.rain_prob:
            x_adv[0] = add_rain(x_adv[0], rain_type=rain_type, normalized=True)

    if args.adv:
        y_pred, x_processed, mask, raw_mask = art_model.predict(x_adv)
    else:
        y_pred, x_processed, mask, raw_mask = art_model.predict(x)

    y_pred = [{k: torch.from_numpy(v) for k, v in t.items()} for t in y_pred]
    res = {y['image_id'].item(): y_pred[0]}
    # res = {target["image_id"].item(): output for target, output in zip(y, y_pred)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()
