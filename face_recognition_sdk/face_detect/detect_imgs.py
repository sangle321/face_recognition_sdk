"""Face detection using UltraFace / MobileNet-SSD style detector."""

import os
import sys

import cv2
import numpy as np
import torch

from face_recognition_sdk.face_detect.vision.ssd.config.fd_config import define_img_size

_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_MODELS_DIR = os.path.join(_PKG_DIR, "models")

INPUT_SIZE = 320
THRESHOLD = 0.6
CANDIDATE_SIZE = 1500
NET_TYPE = "slim"  # "slim" or "RFB"
DEVICE = "cpu"

define_img_size(INPUT_SIZE)

from face_recognition_sdk.face_detect.vision.ssd.mb_tiny_fd import (  # noqa: E402
    create_mb_tiny_fd,
    create_mb_tiny_fd_predictor,
)
from face_recognition_sdk.face_detect.vision.ssd.mb_tiny_RFB_fd import (  # noqa: E402
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)

_label_path = os.path.join(_MODELS_DIR, "voc-model-labels.txt")
_class_names = [name.strip() for name in open(_label_path).readlines()]

if NET_TYPE == "slim":
    _model_path = os.path.join(_MODELS_DIR, "pretrained", "version-slim-320.pth")
    _net = create_mb_tiny_fd(len(_class_names), is_test=True, device=DEVICE)
    _predictor = create_mb_tiny_fd_predictor(
        _net, candidate_size=CANDIDATE_SIZE, device=DEVICE
    )
elif NET_TYPE == "RFB":
    _model_path = os.path.join(_MODELS_DIR, "pretrained", "version-RFB-320.pth")
    _net = create_Mb_Tiny_RFB_fd(len(_class_names), is_test=True, device=DEVICE)
    _predictor = create_Mb_Tiny_RFB_fd_predictor(
        _net, candidate_size=CANDIDATE_SIZE, device=DEVICE
    )
else:
    raise ValueError(f"Unknown net_type: {NET_TYPE}")

_net.load(_model_path)


def get_face_boundingbox(orig_image):
    """
    Detect faces in a BGR image.

    Args:
        orig_image: numpy array (H, W, 3) BGR image.

    Returns:
        boxes: torch.Tensor of shape (N, 4) — [x1, y1, x2, y2]
        probs: torch.Tensor of shape (N,) — confidence scores
    """
    boxes, labels, probs = _predictor.predict(
        cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB),
        CANDIDATE_SIZE / 2,
        THRESHOLD,
    )

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    height, width, _ = orig_image.shape
    valid = (
        (boxes[:, 0] >= 0)
        & (boxes[:, 1] >= 0)
        & (boxes[:, 2] < width)
        & (boxes[:, 3] < height)
    ).bool()
    boxes = boxes[valid]
    probs = probs[valid]
    return boxes, probs
