####### Copied from SegmentAndComplete with minor modifications to add rain #######
####### https://arxiv.org/pdf/2112.04532 #######

from RainPGD.pgd_patch import PGDPatch
from armory import paths
from vision.torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pytorch_faster_rcnn import PyTorchFasterRCNN
from tqdm import tqdm
from coco.coco_utils import get_coco
import presets
import torch
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Generating Adversarial Dataset")
parser.add_argument("--iter", type=int, default=200, help='number of adverserial training iterations')
parser.add_argument("--world_size", type=int, default=1, help='total number of jobs')
parser.add_argument("--rank", type=int, default=0, help='job ID')
parser.add_argument("--patch_size", type=int, default=100)
parser.add_argument("--random", action='store_true', default=False)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--n_imgs", type=int, default=1000, help='number of adv images to generate')
parser.add_argument("--image_set", type=str, default='val')
parser.add_argument("--rain", action='store_true', default=False, help='train for rain robustness')
parser.add_argument("--rain_prob", type=float, default=0.7, help='rain probability')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setup dir
if args.random:
    save_dir = f'adv_data/coco_random_patch_{args.patch_size}'
else:
    save_dir = f'adv_data/coco_topleft_patch_{args.patch_size}'
if args.rain:
    save_dir = f'{save_dir}/rain/{args.image_set}'
else:
    save_dir = f'{save_dir}/normal/{args.image_set}'
data_dir = os.path.join(save_dir, 'data')
img_dir = os.path.join(save_dir, 'image')
losses_dir = os.path.join(save_dir, 'losses')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(losses_dir, exist_ok=True)

# setup model
paths.set_mode("host")
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

attacker = PGDPatch(art_model, rain=args.rain, rain_prob=args.rain_prob, batch_size=1, eps=1.0, eps_step=0.01, max_iter=args.iter, num_random_init=0, random_eps=False,
                    targeted=False, verbose=True)

# setup dataset
dataset = get_coco("./data/", image_set=args.image_set, transforms=presets.DetectionPresetEval())

dataset_size = min(args.n_imgs, len(dataset))


chunk_size = dataset_size // args.world_size
start_ind = args.rank * chunk_size
if args.rank == args.world_size - 1:
    end_ind = dataset_size
else:
    end_ind = (args.rank + 1) * chunk_size

dataset = torch.utils.data.Subset(dataset, list(range(start_ind, end_ind)))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
pbar = tqdm(data_loader)
for i, data in enumerate(pbar):
    x, y = data
    x = x[0].unsqueeze(0).permute(0, 2, 3, 1).numpy()
    label = {k: v.squeeze(0).numpy() for k, v in y.items()}
    patch_height = args.patch_size
    patch_width = args.patch_size
    if args.random:
        h = x.shape[1]
        w = x.shape[2]
        ymin = np.random.randint(0, h - patch_height)
        xmin = np.random.randint(0, w - patch_width)
    else:
        xmin = 0
        ymin = 0
    x_adv, losses = attacker.generate(x, y=label, patch_height=patch_height, patch_width=patch_width, xmin=xmin, ymin=ymin)
    
    losses_fn = os.path.join(losses_dir, f'{i}.png')
    plt.clf()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(losses_fn)

    # save adv img
    img_fn = os.path.join(img_dir, f'{i:06d}.png')
    x_adv_save = cv2.cvtColor(x_adv[0], cv2.COLOR_BGR2RGB)*255
    cv2.imwrite(img_fn, x_adv_save)

    # save data
    data_fn = os.path.join(data_dir, f'{i:06d}.pt')
    torch.save({
        'xmin': xmin,
        'ymin': ymin,
        'x': x,
        'y': y,
        'x_adv': x_adv,
        'patch_size': args.patch_size
    }, data_fn)
    pbar.set_description(f"{save_dir} rank {args.rank}")

