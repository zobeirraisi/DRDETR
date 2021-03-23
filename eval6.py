# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import matplotlib as mpl
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops, Rbox_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import time
import sys
import cv2


import os
import time
import argparse
import glob
import shutil
from tqdm import tqdm
import cv2
import torch
import numpy as np


from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list
import torchvision.transforms as T
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from util import Rbox_ops, box_ops


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    parser.add_argument(
        "--dataset", default="data/44015", type=str, help="folder path to input images",
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )
    parser.add_argument(
        "--img_path",
        default="data/44015/img_1.jpg",
        type=str,
        help="input image file for inference",
    )
    parser.add_argument(
        "--resume", default="out/beluga62.pth", help="resume from checkpoint"
    )
    return parser


# standard PyTorch mean-std input image normalization
transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h, c, s = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h), c, s]
    return torch.stack(b, dim=1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1, c, s = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0), c, s]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h, 1, 1], dtype=torch.float32)
    b = box_xyxy_to_cxcywh(b)
    return b


def detect(im, model, transform):

    im_h, im_w = im.size
    img = transform(im).unsqueeze(0).cuda()
    outputs = model(img)

    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    # print(out_bbox)

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), 100, dim=1
    )
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 6))
    keep = scores[0] > 0.25
    boxes = boxes[0, keep]
    labels = labels[0, keep]
    one = torch.tensor([1]).cuda()
    target_sizes = torch.tensor([[im_w, im_h]])
    target_sizes = target_sizes.cuda()
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h, one, one], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    boxes2 = box_ops.box_xyxy_to_cxcywh(boxes)
    boxes_corners = Rbox_ops.box_center_to_corners(boxes2).squeeze(0)

    return boxes_corners


@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # t0 = time.time()

    Path = args.dataset
    # print(Path)
    folder = Path.split("/")[-1]

    image_files = glob.glob(os.path.join(Path, "*.jpg"))
    print(image_files)
    for im in tqdm(image_files, total=len(image_files)):
        name = im.split("/")[-1]
        img = Image.open(im).convert("RGB")
        boxes = detect(img, model, transform)
        # print(boxes)
        im2 = img.copy()
        draw = ImageDraw.Draw(im2)
        for (x1, y1, x2, y2, x3, y3, x4, y4) in boxes.tolist():
            draw.polygon([x1, y1, x2, y2, x3, y3, x4, y4], None, "red")
        name2 = name.split(".")[0]
        textfile = f"OUTPUT/txt-{folder}/res_{name2}.txt"
        os.makedirs(f"OUTPUT/txt-{folder}", exist_ok=True)
        with open(textfile, "w") as f:
            for (x1, y1, x2, y2, x3, y3, x4, y4) in boxes.tolist():
                # draw.rectangle((xmin, ymin, xmax, ymax), None, "cyan", width=2)
                # txt = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)},{int(x3)},{int(y3)},{int(x4)},{int(y4)}\n"
                txt = f"{int(x2)},{int(y2)},{int(x3)},{int(y3)},{int(x4)},{int(y4)},{int(x1)},{int(y1)}\n"
                f.write(txt)

        os.makedirs(f"OUTPUT/imgs-{folder}", exist_ok=True)
        im2.save(f"OUTPUT/imgs-{folder}/res_{name}", "JPEG")

    shutil.make_archive("predict", "zip", f"OUTPUT/txt-{folder}")
    os.system(f"python eval/script.py -g=eval/ATS_GT/{folder}.zip -s=predict.zip")
    # results = [
    #     {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)
    # ]
    # print("Outputs", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
