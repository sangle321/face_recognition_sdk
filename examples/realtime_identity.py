#!/usr/bin/env python3
"""
Real-time Face Identity Application
====================================
- Live webcam face detection + recognition
- Enroll known people (from images or live capture)
- 1:N matching against a local gallery
- On-screen labels: name + similarity score

Controls (while camera window is focused):
  E          Enroll the largest face currently visible (type name in terminal)
  L          Load / reload gallery from disk
  S          Save current gallery to disk
  C          Clear gallery
  Q / ESC    Quit

Usage:
  python examples/realtime_identity.py
  python examples/realtime_identity.py --camera 0 --threshold 75
  python examples/realtime_identity.py --gallery gallery.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow running without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_recognition_sdk import FaceRecognition


# ---------------------------------------------------------------------------
# Gallery (in-memory + optional disk persistence)
# ---------------------------------------------------------------------------

class FaceGallery:
    """Simple 1:N face gallery stored as name → list of embeddings."""

    def __init__(self, threshold: float = 75.0):
        self.threshold = threshold
        # name -> list of (embedding ndarray)
        self._db: Dict[str, List[np.ndarray]] = {}

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be empty")
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        self._db.setdefault(name, []).append(emb)

    def remove(self, name: str) -> bool:
        return self._db.pop(name, None) is not None

    def clear(self) -> None:
        self._db.clear()

    def names(self) -> List[str]:
        return sorted(self._db.keys())

    def count(self) -> int:
        return sum(len(v) for v in self._db.values())

    def identify(
        self, embedding: np.ndarray, sdk: FaceRecognition
    ) -> Tuple[Optional[str], float]:
        """
        Return (best_name, best_score).
        If no match above threshold, returns (None, best_score).
        """
        if not self._db:
            return None, 0.0

        emb = np.asarray(embedding, dtype=np.float32).ravel()
        best_name: Optional[str] = None
        best_score = -1.0

        for name, embs in self._db.items():
            for ref in embs:
                score = sdk.get_similarity(emb, ref)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < self.threshold:
            return None, best_score
        return best_name, best_score

    def save(self, path: str) -> None:
        names, vectors = [], []
        for name, embs in self._db.items():
            for e in embs:
                names.append(name)
                vectors.append(e)
        if not vectors:
            # save empty structure
            np.savez_compressed(path, names=np.array([], dtype=object), vectors=np.zeros((0, 1)))
        else:
            np.savez_compressed(
                path,
                names=np.array(names, dtype=object),
                vectors=np.stack(vectors, axis=0),
            )
        print(f"[gallery] saved {len(vectors)} embedding(s) → {path}")

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"[gallery] file not found: {path}")
            return
        data = np.load(path, allow_pickle=True)
        names = data["names"]
        vectors = data["vectors"]
        self._db.clear()
        for name, vec in zip(names, vectors):
            self._db.setdefault(str(name), []).append(np.asarray(vec, dtype=np.float32))
        print(f"[gallery] loaded {len(names)} embedding(s) from {path}")
        print(f"[gallery] identities: {self.names()}")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_face(
    frame: np.ndarray,
    bbox: np.ndarray,
    label: str,
    score: float,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} ({score:.0f})" if score > 0 else label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # background bar
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th - 10)),
        (x1 + tw + 6, y1),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 3, max(th + 2, y1 - 6)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, fps: float, gallery: FaceGallery) -> None:
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Gallery: {len(gallery.names())} people / {gallery.count()} embeddings",
        "E=Enroll  L=Load  S=Save  C=Clear  Q=Quit",
    ]
    y = 22
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time Face Identity")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
    p.add_argument("--threshold", type=float, default=75.0, help="Match threshold 0-100 (default 75)")
    p.add_argument("--gallery", type=str, default="gallery.npz", help="Gallery file path")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument(
        "--enroll-dir",
        type=str,
        default=None,
        help="Optional: pre-enroll all images in this folder (subfolder name = person name)",
    )
    p.add_argument("--max-faces", type=int, default=5, help="Max faces to process per frame")
    return p.parse_args()


def enroll_from_directory(sdk: FaceRecognition, gallery: FaceGallery, root: str) -> None:
    """
    Directory layout:
        enroll_dir/
          Alice/
            img1.jpg
            img2.jpg
          Bob/
            photo.png
    """
    if not os.path.isdir(root):
        print(f"[enroll] directory not found: {root}")
        return
    for person in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path = os.path.join(person_dir, fname)
            try:
                faces = sdk.get_image_info(path, face_max_count=1)
                if not faces:
                    print(f"[enroll] no face in {path}")
                    continue
                gallery.enroll(person, faces[0]["embedding"])
                print(f"[enroll] {person} ← {fname}")
            except Exception as e:
                print(f"[enroll] failed {path}: {e}")
    print(f"[enroll] done. Identities: {gallery.names()}")


def main():
    args = parse_args()

    print("Loading Face Recognition SDK (models on CPU)...")
    sdk = FaceRecognition(threshold=args.threshold)
    gallery = FaceGallery(threshold=args.threshold)

    # Load existing gallery if present
    if os.path.isfile(args.gallery):
        gallery.load(args.gallery)

    # Optional bulk enroll from folder structure
    if args.enroll_dir:
        enroll_from_directory(sdk, gallery, args.enroll_dir)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera}")
        print("Tip: try --camera 1  or check permissions / other apps using the camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("\n=== Real-time Face Identity ===")
    print("Window controls (focus the camera window):")
    print("  E   Enroll current largest face (enter name in this terminal)")
    print("  L   Reload gallery from disk")
    print("  S   Save gallery to disk")
    print("  C   Clear gallery")
    print("  Q   Quit")
    print("--------------------------------\n")

    window = "Face Identity"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps = 0.0
    t_prev = time.time()
    frame_count = 0

    # Process every N-th frame for speed (detection is the bottleneck)
    PROCESS_EVERY = 2
    last_results: List[dict] = []  # cached results for skipped frames

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_count += 1
            process_now = (frame_count % PROCESS_EVERY == 0) or not last_results

            if process_now:
                faces = sdk.get_image_info(frame, face_max_count=args.max_faces)
                results = []
                for face in faces:
                    name, score = gallery.identify(face["embedding"], sdk)
                    results.append(
                        {
                            "bbox": face["bbox"],
                            "name": name,
                            "score": score,
                            "embedding": face["embedding"],
                        }
                    )
                last_results = results

            # Draw
            for r in last_results:
                if r["name"]:
                    color = (0, 200, 0)  # green = known
                    label = r["name"]
                else:
                    color = (0, 0, 220)  # red = unknown
                    label = "Unknown"
                draw_face(frame, r["bbox"], label, r["score"], color)

            # FPS
            t_now = time.time()
            dt = t_now - t_prev
            if dt >= 0.5:
                fps = frame_count / dt
                frame_count = 0
                t_prev = t_now

            draw_hud(frame, fps, gallery)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break

            elif key in (ord("e"), ord("E")):
                if not last_results:
                    print("[enroll] no face detected — look at the camera")
                    continue
                # Enroll the largest face (by bbox area)
                largest = max(
                    last_results,
                    key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
                )
                try:
                    name = input("Enter name for this face: ").strip()
                except EOFError:
                    name = ""
                if name:
                    gallery.enroll(name, largest["embedding"])
                    print(f"[enroll] '{name}' added. Gallery: {gallery.names()}")
                else:
                    print("[enroll] cancelled (empty name)")

            elif key in (ord("s"), ord("S")):
                gallery.save(args.gallery)

            elif key in (ord("l"), ord("L")):
                gallery.load(args.gallery)

            elif key in (ord("c"), ord("C")):
                gallery.clear()
                print("[gallery] cleared")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
