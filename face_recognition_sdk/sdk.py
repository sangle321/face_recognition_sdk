"""
High-level Face Recognition SDK API.

Provides a clean object-oriented interface over the underlying
detection / landmark / pose / feature modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from face_recognition_sdk.face_detect.detect_imgs import get_face_boundingbox
from face_recognition_sdk.face_feature.GetFeature import get_face_feature
from face_recognition_sdk.face_landmark.GetLandmark import get_face_landmark
from face_recognition_sdk.face_pose.GetPose import get_face_pose


class FaceRecognition:
    """
    On-premise face recognition engine.

    Capabilities
    ------------
    - Face detection (bounding boxes + confidence)
    - 68-point facial landmarks
    - Head pose estimation (yaw / pitch / roll)
    - Feature embedding extraction
    - Cosine-similarity based face comparison

    Example
    -------
    >>> sdk = FaceRecognition()
    >>> faces = sdk.get_image_info("photo.jpg", face_max_count=5)
    >>> if faces:
    ...     print(faces[0]["bbox"], faces[0]["embedding"].shape)
    >>> score = sdk.get_similarity(faces[0]["embedding"], other_embedding)
    """

    DEFAULT_THRESHOLD = 75.0  # similarity score threshold for "same person"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Args:
            threshold: Similarity score (0–100) above which two faces
                       are considered the same person. Default is 75.
        """
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_image_info(
        self,
        image: Union[str, np.ndarray],
        face_max_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Detect faces and extract full information for each face.

        Args:
            image: Path to an image file **or** a BGR numpy array (H, W, 3).
            face_max_count: Maximum number of faces to return (sorted by
                            detector confidence, highest first).

        Returns:
            List of dicts, one per detected face. Each dict contains:

            - ``bbox``       : np.ndarray shape (4,) — [x1, y1, x2, y2]
            - ``score``      : float — detection confidence
            - ``landmarks``  : np.ndarray shape (136,) — 68 (x,y) pairs flattened
            - ``pose``       : np.ndarray shape (3,) — [rx, ry, rz]
            - ``aligned``    : np.ndarray shape (128, 128, 3) — aligned face crop
            - ``embedding``  : np.ndarray — L2-normalized feature vector
        """
        img = self._load_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        boxes, scores = get_face_boundingbox(img)
        if len(boxes) == 0:
            return []

        # Keep only the top-K faces
        boxes = boxes[:face_max_count]
        scores = scores[:face_max_count]
        count = len(boxes)

        results: List[Dict[str, Any]] = []
        for idx in range(count):
            box = boxes[idx]
            score = float(scores[idx].data.numpy()) if hasattr(scores[idx], "data") else float(scores[idx])

            landmark = get_face_landmark(gray, box).data.numpy()
            pose = get_face_pose(box, landmark)
            aligned, embedding = get_face_feature(img, landmark)

            results.append(
                {
                    "bbox": box.data.numpy() if hasattr(box, "data") else np.asarray(box),
                    "score": score,
                    "landmarks": landmark,
                    "pose": pose,
                    "aligned": aligned,
                    "embedding": embedding,
                }
            )
        return results

    def get_similarity(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
    ) -> float:
        """
        Compute similarity between two face embeddings.

        The score is mapped to the range [0, 100]:

            score = (dot(f1, f2) + 1) * 50

        Args:
            feature1: First embedding (1-D array).
            feature2: Second embedding (1-D array).

        Returns:
            Similarity score in [0, 100]. Higher means more similar.
        """
        f1 = np.asarray(feature1, dtype=np.float64).ravel()
        f2 = np.asarray(feature2, dtype=np.float64).ravel()
        return float((np.sum(f1 * f2) + 1.0) * 50.0)

    def is_same_person(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Decide whether two embeddings belong to the same person.

        Args:
            feature1 / feature2: Face embeddings.
            threshold: Override the instance threshold if provided.

        Returns:
            True if similarity >= threshold.
        """
        thr = self.threshold if threshold is None else float(threshold)
        return self.get_similarity(feature1, feature2) >= thr

    def compare_images(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        face_max_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Convenience method: detect the primary face in each image and compare.

        Returns a dict with keys:
            - ``similarity``   : float
            - ``same_person``  : bool
            - ``faces1``       : list of face dicts from image1
            - ``faces2``       : list of face dicts from image2
        """
        faces1 = self.get_image_info(image1, face_max_count=face_max_count)
        faces2 = self.get_image_info(image2, face_max_count=face_max_count)

        result: Dict[str, Any] = {
            "faces1": faces1,
            "faces2": faces2,
            "similarity": None,
            "same_person": False,
        }

        if faces1 and faces2:
            sim = self.get_similarity(faces1[0]["embedding"], faces2[0]["embedding"])
            result["similarity"] = sim
            result["same_person"] = sim >= self.threshold

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            return img
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Image array must be H×W×3 BGR")
            return image
        raise TypeError("image must be a file path (str) or a numpy BGR array")
