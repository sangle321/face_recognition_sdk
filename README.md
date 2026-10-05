# Face Recognition SDK (Python)

**On-premise Open Source Face Recognition SDK** — face detection, 68-point landmarks, head pose, feature embedding, and 1:1 face matching.

Built on top of the [Faceplugin Open-Source Face Recognition SDK](https://github.com/Faceplugin-ltd/Open-Source-Face-Recognition-SDK).

All processing runs **locally**. No images or embeddings leave your machine.

---

## Features

| Capability              | Description                                      |
|-------------------------|--------------------------------------------------|
| Face Detection          | Bounding boxes + confidence scores               |
| Facial Landmarks        | 68-point landmark model                          |
| Head Pose               | Yaw / pitch / roll estimation                    |
| Feature Embedding       | L2-normalized deep embedding (IRN-50)            |
| Face Matching           | Cosine-similarity score mapped to 0–100          |
| Platforms               | Linux & Windows (CPU; GPU optional via PyTorch)  |

---

## Requirements

- **Python** 3.9+
- **Linux** or **Windows**
- Dependencies: `numpy`, `opencv-python`, `Pillow`, `torch`, `torchvision`

> The native alignment/pose library (`libFaceUtil.so` / `FaceUtil.dll`) is included.

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Or using conda
conda create -n facesdk python=3.10
conda activate facesdk
```

### 2. Install the package

```bash
# From the project root
pip install -r requirements.txt
pip install -e .
```

### 3. Quick test

```bash
python examples/compare_faces.py
```

You should see a similarity score and a same-person decision for the two sample images in `data/`.

---

## Quick Start

```python
from face_recognition_sdk import FaceRecognition

sdk = FaceRecognition(threshold=75)   # default threshold

# Full pipeline on one image
faces = sdk.get_image_info("photo.jpg", face_max_count=5)

for i, face in enumerate(faces):
    print(f"Face {i}:")
    print(f"  bbox      = {face['bbox']}")       # [x1, y1, x2, y2]
    print(f"  score     = {face['score']:.3f}")
    print(f"  landmarks = shape {face['landmarks'].shape}")  # (136,)
    print(f"  pose      = {face['pose']}")       # [rx, ry, rz]
    print(f"  embedding = shape {face['embedding'].shape}")

# Compare two embeddings
if len(faces) >= 2:
    score = sdk.get_similarity(faces[0]["embedding"], faces[1]["embedding"])
    same  = sdk.is_same_person(faces[0]["embedding"], faces[1]["embedding"])
    print(f"Similarity = {score:.2f}, same person = {same}")
```

### Compare two images directly

```python
result = sdk.compare_images("person_a.jpg", "person_b.jpg")
print(result["similarity"], result["same_person"])
```

---

## API Reference

### `FaceRecognition(threshold=75.0)`

| Method | Description |
|--------|-------------|
| `get_image_info(image, face_max_count=10)` | Detect faces and return list of dicts with `bbox`, `score`, `landmarks`, `pose`, `aligned`, `embedding`. `image` can be a file path or a BGR `numpy` array. |
| `get_similarity(feat1, feat2)` | Returns similarity score in **[0, 100]**. |
| `is_same_person(feat1, feat2, threshold=None)` | `True` if score ≥ threshold. |
| `compare_images(image1, image2, face_max_count=1)` | Convenience wrapper that returns similarity + both face lists. |

**Similarity formula:**  
`score = (dot(f1, f2) + 1) * 50`  
(where embeddings are L2-normalized → dot product ∈ [-1, 1])

**Default decision threshold:** 75

---

## Project Structure

```
face_recognition_sdk/
├── face_recognition_sdk/          # Installable package
│   ├── __init__.py
│   ├── sdk.py                     # High-level FaceRecognition class
│   ├── face_detect/               # UltraFace-style detector + models
│   ├── face_landmark/             # 68-pt landmark model
│   ├── face_feature/              # IRN-50 embedding model
│   ├── face_pose/                 # Pose estimation
│   └── face_util/                 # Native lib (align + pose)
│       └── C/
│           ├── libFaceUtil.so     # Linux
│           └── FaceUtil.dll       # Windows
├── data/                          # Sample images (1.jpg, 2.png)
├── examples/
│   ├── compare_faces.py
│   └── detect_faces.py
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Examples

```bash
# Compare the two bundled test images
python examples/compare_faces.py

# Compare your own images
python examples/compare_faces.py path/to/a.jpg path/to/b.jpg --threshold 70

# Detect faces and save visualization
python examples/detect_faces.py path/to/photo.jpg -o result.jpg
```

---

## Notes

- Models run on **CPU** by default. To use GPU, ensure a CUDA-enabled PyTorch is installed and change `DEVICE = "cuda"` in `face_detect/detect_imgs.py` (and move models to GPU if desired).
- Relative paths inside the original upstream code have been converted to **package-relative absolute paths**, so the SDK works correctly after `pip install -e .`.
- This open-source release has lower accuracy than Faceplugin’s commercial SDKs. Contact [info@faceplugin.com](mailto:info@faceplugin.com) for production-grade engines (liveness, higher FRVT accuracy, mobile SDKs, etc.).

---

## License & Attribution

Original models and core algorithms © Faceplugin.  
This packaging and high-level API are provided for convenient integration.

Website: [https://faceplugin.com](https://faceplugin.com)
