#!/usr/bin/env python3
"""
Batch-enroll faces from a folder structure into the gallery.

Directory layout:
  enroll/
    Alice/
      photo1.jpg
      photo2.png
    Bob/
      face.jpg

Usage:
  python examples/enroll_from_images.py --input enroll --output gallery.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_recognition_sdk import FaceRecognition
from realtime_identity import FaceGallery  # reuse gallery class


def main():
    parser = argparse.ArgumentParser(description="Batch enroll faces into gallery")
    parser.add_argument("--input", "-i", required=True, help="Root folder (subfolder = person name)")
    parser.add_argument("--output", "-o", default="gallery.npz", help="Output gallery file")
    parser.add_argument("--threshold", type=float, default=75.0)
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: folder not found: {args.input}")
        sys.exit(1)

    sdk = FaceRecognition(threshold=args.threshold)
    gallery = FaceGallery(threshold=args.threshold)

    for person in sorted(os.listdir(args.input)):
        person_dir = os.path.join(args.input, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            path = os.path.join(person_dir, fname)
            faces = sdk.get_image_info(path, face_max_count=1)
            if not faces:
                print(f"  skip (no face): {path}")
                continue
            gallery.enroll(person, faces[0]["embedding"])
            print(f"  enrolled {person} ← {fname}")

    gallery.save(args.output)
    print(f"\nDone. Identities: {gallery.names()}")
    print(f"Total embeddings: {gallery.count()}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
