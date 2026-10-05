"""Face feature (embedding) extraction using IRN-50."""

import ctypes
import os

import cv2
import numpy as np
import torch

from face_recognition_sdk.face_feature.irn50_pytorch import irn50_pytorch
from face_recognition_sdk.face_util.faceutil import align_vertical

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_WEIGHTS = os.path.join(_PKG_DIR, "irn50_pytorch.npy")

_model = irn50_pytorch(_WEIGHTS)
_model.eval()

_feature_align_image = np.zeros([128, 128, 3], dtype=np.uint8)


def get_face_feature(image, landmark):
    """
    Align face and extract L2-normalized embedding.

    Args:
        image: BGR image (H, W, 3)
        landmark: flattened 136-element landmark array

    Returns:
        align_img: aligned face crop (128, 128, 3) uint8
        feature: 1-D numpy array (embedding dimension)
    """
    landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    align_vertical(
        image,
        image.shape[1],
        image.shape[0],
        _feature_align_image,
        128,
        128,
        3,
        landmark_vec,
        48,
        64,
        40,
    )

    proc = _feature_align_image / 256.0
    proc = torch.from_numpy(proc.astype(np.float32))
    proc = proc.permute(2, 0, 1).unsqueeze(0)

    feature_out = _model(proc)
    feature_out = torch.nn.functional.normalize(feature_out)[0, :]
    return _feature_align_image.copy(), feature_out.data.numpy()
