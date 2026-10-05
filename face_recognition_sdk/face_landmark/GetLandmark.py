"""68-point facial landmark detection."""

import os

import cv2
import numpy as np
import torch

from face_recognition_sdk.face_landmark.MobileFaceNet import MobileFaceNet

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_WEIGHTS = os.path.join(_PKG_DIR, "vfl_1.02_578_6.734591484069824.pth.tar")

_model = MobileFaceNet([64, 64], 136)
_state = torch.load(_WEIGHTS, map_location=torch.device("cpu"), weights_only=False)
_model.load_state_dict(_state["state_dict"])
_model.eval()


def get_face_landmark(gray_img, bounding_box):
    """
    Extract 68 facial landmarks for a single face.

    Args:
        gray_img: grayscale image (H, W)
        bounding_box: [x1, y1, x2, y2] (tensor or array)

    Returns:
        landmark: 1-D tensor of shape (136,) — flattened (x,y) pairs
    """
    image = gray_img
    box = bounding_box

    nHeight, nWidth = image.shape

    rLeftMargin = 0.05
    rTopMargin = 0.00
    rRightMargin = 0.05
    rBottomMargin = 0.10

    rW = box[2] - box[0]
    rH = box[3] - box[1]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    sz = pow(rW * rH, 0.5)
    rX = cx - sz / 2
    rY = cy - sz / 2
    rW = sz
    rH = sz

    iExFaceX = int(rX - rLeftMargin * rW)
    iExFaceY = int(rY - rTopMargin * rH)
    iExFaceW = int((1 + (rLeftMargin + rRightMargin)) * rW)
    iExFaceH = int((1 + (rTopMargin + rBottomMargin)) * rH)

    iExFaceX = np.clip(iExFaceX, 0, nWidth - 1)
    iExFaceY = np.clip(iExFaceY, 0, nHeight - 1)
    iExFaceW = np.clip(iExFaceX + iExFaceW, 0, nWidth - 1) - iExFaceX
    iExFaceH = np.clip(iExFaceY + iExFaceH, 0, nHeight - 1) - iExFaceY

    image = image[iExFaceY : iExFaceY + iExFaceH, iExFaceX : iExFaceX + iExFaceW]
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    image = image / 256.0
    image = torch.from_numpy(image.astype(np.float32))
    image = image.unsqueeze(0).unsqueeze(0)

    landmark = _model(image)
    landmark = landmark.reshape(68, 2)
    landmark[:, 0] = landmark[:, 0] * iExFaceW + iExFaceX
    landmark[:, 1] = landmark[:, 1] * iExFaceH + iExFaceY
    landmark = landmark.reshape(-1)

    return landmark
