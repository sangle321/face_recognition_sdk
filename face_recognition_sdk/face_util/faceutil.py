"""Native face utility library bindings (alignment + pose)."""

import ctypes
import os
import sys
from numpy.ctypeslib import ndpointer

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))

if sys.platform.startswith("linux"):
    _dll_path = os.path.join(_PKG_DIR, "C", "libFaceUtil.so")
    face_util_engine = ctypes.cdll.LoadLibrary(_dll_path)
elif sys.platform.startswith("win32"):
    _dll_path = os.path.join(_PKG_DIR, "C", "FaceUtil.dll")
    face_util_engine = ctypes.CDLL(_dll_path)
else:
    raise RuntimeError(
        f"Unsupported platform: {sys.platform}. "
        "Only Linux and Windows are supported."
    )

align_vertical = face_util_engine.align_vertical
align_vertical.argtypes = [
    ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
align_vertical.restype = ctypes.c_int32

getPose = face_util_engine.getPose
getPose.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
getPose.restype = ctypes.c_int32

getPose68 = face_util_engine.getPose68
getPose68.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
getPose68.restype = ctypes.c_int32
