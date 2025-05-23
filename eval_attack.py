from coco.coco_utils import get_coco_using_paths, get_coco_api_from_dataset
from coco.coco_eval import CocoEvaluator
from vision.torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pytorch_faster_rcnn import PyTorchFasterRCNN
from RainPGD.utils import add_rain
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            x[0] = add_rain(x[0], rain_type=rain_type, normalized=True)

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

# Compute per-class metrics and save them into a CSV.
coco_eval = coco_evaluator.coco_eval['bbox']
precision = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
recall = coco_eval.eval['recall']        # shape: [T, K, A, M]

# Get the current IoU thresholds and find indices
iou_thrs = coco_eval.params.iouThrs
idx_50 = np.where(np.abs(iou_thrs - 0.5) < 1e-5)[0][0]
idx_75 = np.where(np.abs(iou_thrs - 0.75) < 1e-5)[0][0]

# Get maxDets indices
max_dets = coco_eval.params.maxDets
idx_1 = 0  # Index for maxDets=1
idx_10 = 1  # Index for maxDets=10
idx_100 = 2  # Index for maxDets=100
idx_max = -1  # Last index (for whatever the max setting is, often 100)

cat_ids = coco_eval.params.catIds
categories = coco_eval.cocoGt.loadCats(cat_ids)
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

results = []

for idx, catId in enumerate(cat_ids):
    cat_name = cat_id_to_name.get(catId, str(catId))
    
    # Compute AP metrics
    # AP across all IoUs (averaged)
    prec_all = precision[:, :, idx, 0, idx_max]  # "all" area, max detections
    valid_all = prec_all > -1
    ap_all = np.mean(prec_all[valid_all]) if np.any(valid_all) else float('nan')
    
    # AP at IoU=0.5
    prec_50 = precision[idx_50, :, idx, 0, idx_max]
    valid_50 = prec_50 > -1
    ap_50 = np.mean(prec_50[valid_50]) if np.any(valid_50) else float('nan')
    
    # AP at IoU=0.75
    prec_75 = precision[idx_75, :, idx, 0, idx_max]
    valid_75 = prec_75 > -1
    ap_75 = np.mean(prec_75[valid_75]) if np.any(valid_75) else float('nan')
    
    # AP for small, medium, and large objects (across all IoUs)
    prec_small = precision[:, :, idx, 1, idx_max]
    valid_small = prec_small > -1
    ap_small = np.mean(prec_small[valid_small]) if np.any(valid_small) else float('nan')
    
    prec_medium = precision[:, :, idx, 2, idx_max]
    valid_medium = prec_medium > -1
    ap_medium = np.mean(prec_medium[valid_medium]) if np.any(valid_medium) else float('nan')
    
    prec_large = precision[:, :, idx, 3, idx_max]
    valid_large = prec_large > -1
    ap_large = np.mean(prec_large[valid_large]) if np.any(valid_large) else float('nan')
    
    # AR metrics for different maxDets
    rec_all_1 = recall[:, idx, 0, idx_1]
    valid_rec_all_1 = rec_all_1 > -1
    ar_all_1 = np.mean(rec_all_1[valid_rec_all_1]) if np.any(valid_rec_all_1) else float('nan')
    
    rec_all_10 = recall[:, idx, 0, idx_10]
    valid_rec_all_10 = rec_all_10 > -1
    ar_all_10 = np.mean(rec_all_10[valid_rec_all_10]) if np.any(valid_rec_all_10) else float('nan')
    
    rec_all_100 = recall[:, idx, 0, idx_100]
    valid_rec_all_100 = rec_all_100 > -1
    ar_all_100 = np.mean(rec_all_100[valid_rec_all_100]) if np.any(valid_rec_all_100) else float('nan')
    
    # AR metrics for small, medium, and large objects
    rec_small = recall[:, idx, 1, idx_max]
    valid_rec_small = rec_small > -1
    ar_small = np.mean(rec_small[valid_rec_small]) if np.any(valid_rec_small) else float('nan')
    
    rec_medium = recall[:, idx, 2, idx_max]
    valid_rec_medium = rec_medium > -1
    ar_medium = np.mean(rec_medium[valid_rec_medium]) if np.any(valid_rec_medium) else float('nan')
    
    rec_large = recall[:, idx, 3, idx_max]
    valid_rec_large = rec_large > -1
    ar_large = np.mean(rec_large[valid_rec_large]) if np.any(valid_rec_large) else float('nan')
    
    results.append({
        'Class': cat_name,
        'AP_all': ap_all,
        'AP_50': ap_50,
        'AP_75': ap_75,
        'AP_small': ap_small,
        'AP_medium': ap_medium,
        'AP_large': ap_large,
        'AR_all_1': ar_all_1,
        'AR_all_10': ar_all_10,
        'AR_all_100': ar_all_100,
        'AR_small': ar_small,
        'AR_medium': ar_medium,
        'AR_large': ar_large
    })

# Add the overall statistics for comparison
stats = coco_eval.stats
total_metrics = {
    'Class': 'Total',
    'AP_all': stats[0],
    'AP_50': stats[1],
    'AP_75': stats[2],
    'AP_small': stats[3],
    'AP_medium': stats[4],
    'AP_large': stats[5],
    'AR_all_1': stats[6],
    'AR_all_10': stats[7],
    'AR_all_100': stats[8],
    'AR_small': stats[9],
    'AR_medium': stats[10],
    'AR_large': stats[11]
}
results.append(total_metrics)

# Save to CSV
eval_type = "normal"
if args.adv:
    eval_type = "adversarial"
if args.rain_adv:
    eval_type = "rain_adversarial"
elif args.rain:
    eval_type = f"rain_{args.rain_prob}"

folder_name = os.path.basename(args.data_folder)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

output_path = f"results/{folder_name}_{eval_type}_{timestamp}.csv"
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"Saved per-class metrics to {output_path}")