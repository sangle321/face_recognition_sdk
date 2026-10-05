"""Face pose (yaw, pitch, roll) estimation from landmarks."""

import ctypes

import numpy as np

from face_recognition_sdk.face_util.faceutil import getPose68


def get_face_pose(box, landmark):
    """
    Estimate head pose from bounding box and 68 landmarks.

    Args:
        box: [x1, y1, x2, y2]
        landmark: flattened 136-element array/list of (x,y) pairs

    Returns:
        np.ndarray of shape (3,) — [rx, ry, rz] (radians-ish / degrees depending on lib)
    """
    fw = box[2] - box[0]
    fh = box[3] - box[1]
    sz = pow(fw * fh, 0.5)
    cx = (box[2] + box[0]) / 2
    cy = (box[3] + box[1]) / 2
    x1 = cx - sz / 2
    y1 = cy - sz / 2

    rx = ctypes.c_float(0)
    ry = ctypes.c_float(0)
    rz = ctypes.c_float(0)

    landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    getPose68(
        x1,
        y1,
        sz,
        landmark_vec,
        68,
        ctypes.pointer(rx),
        ctypes.pointer(ry),
        ctypes.pointer(rz),
    )
    return np.array([rx.value, ry.value, rz.value], dtype=np.float32)
