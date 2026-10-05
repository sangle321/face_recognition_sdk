#!/usr/bin/env python3
"""
Example: Compare two face images and print similarity.
"""

import argparse
import os
import sys

# Allow running without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_recognition_sdk import FaceRecognition


def main():
    parser = argparse.ArgumentParser(description="Compare two face images")
    parser.add_argument("image1", nargs="?", default=None, help="Path to first image")
    parser.add_argument("image2", nargs="?", default=None, help="Path to second image")
    parser.add_argument(
        "--threshold", type=float, default=75.0, help="Similarity threshold (default 75)"
    )
    args = parser.parse_args()

    # Default test images shipped with the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img1 = args.image1 or os.path.join(project_root, "data", "1.jpg")
    img2 = args.image2 or os.path.join(project_root, "data", "2.png")

    if not os.path.isfile(img1):
        print(f"Error: image not found: {img1}")
        sys.exit(1)
    if not os.path.isfile(img2):
        print(f"Error: image not found: {img2}")
        sys.exit(1)

    print(f"Image 1 : {img1}")
    print(f"Image 2 : {img2}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)

    sdk = FaceRecognition(threshold=args.threshold)
    result = sdk.compare_images(img1, img2, face_max_count=1)

    if not result["faces1"]:
        print("No face detected in image 1")
        sys.exit(1)
    if not result["faces2"]:
        print("No face detected in image 2")
        sys.exit(1)

    face1 = result["faces1"][0]
    face2 = result["faces2"][0]

    print(f"Face 1 bbox  : {face1['bbox']}")
    print(f"Face 1 score : {face1['score']:.3f}")
    print(f"Face 1 pose  : {face1['pose']}")
    print(f"Face 2 bbox  : {face2['bbox']}")
    print(f"Face 2 score : {face2['score']:.3f}")
    print(f"Face 2 pose  : {face2['pose']}")
    print("-" * 50)
    print(f"Similarity   : {result['similarity']:.2f}")
    print(f"Same person  : {result['same_person']}")


if __name__ == "__main__":
    main()
