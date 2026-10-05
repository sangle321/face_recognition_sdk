#!/usr/bin/env python3
"""
Example: Detect faces in an image and draw bounding boxes + landmarks.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_recognition_sdk import FaceRecognition


def draw_results(image: np.ndarray, faces: list) -> np.ndarray:
    vis = image.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"#{i} {face['score']:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        # Draw 68 landmarks
        lm = face["landmarks"].reshape(68, 2)
        for (x, y) in lm.astype(int):
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Detect faces and landmarks")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image")
    parser.add_argument(
        "-o", "--output", default="output_detect.jpg", help="Output visualization path"
    )
    parser.add_argument("--max-faces", type=int, default=10, help="Max faces to detect")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_path = args.image or os.path.join(project_root, "data", "1.jpg")

    if not os.path.isfile(img_path):
        print(f"Error: image not found: {img_path}")
        sys.exit(1)

    sdk = FaceRecognition()
    faces = sdk.get_image_info(img_path, face_max_count=args.max_faces)

    print(f"Detected {len(faces)} face(s) in {img_path}")
    for i, f in enumerate(faces):
        print(
            f"  [{i}] bbox={f['bbox'].round(1)}, score={f['score']:.3f}, "
            f"pose={f['pose'].round(2)}, embedding_dim={f['embedding'].shape[0]}"
        )

    if faces:
        img = cv2.imread(img_path)
        vis = draw_results(img, faces)
        cv2.imwrite(args.output, vis)
        print(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
