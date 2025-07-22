# Face Recognition with YOLOv8 and ArcFace 🚀

A real-time face recognition system that combines [YOLOv8-face](https://github.com/derronqi/yolov8-face) for face detection and [ArcFace](https://github.com/deepinsight/insightface) for robust face recognition. This project processes videos or webcam input and identifies known individuals with high accuracy.

## 🎥 Demo

![Demo](results/short_output_video.gif)

👉 **Watch the full output video here:** [results/output_video.mp4](results/output_video.mp4)

📌 **Note:** In the demo video, the system detects and recognizes 3 individuals from the face database:
- `Seong_Gi_Hun`
- `Front_Man`
- `Director`

## 🔥 Features
- Real-time face detection using YOLOv8-face
- Identity recognition using ArcFace
- Save output video with bounding boxes and identity overlay
- Easy to extend or use in other projects

## 🛠️ Installation

### 1. Clone this repo

```bash
git clone https://github.com/chinhnguyen25062001/Face_Recognition.git
cd Face_Recognition
```

### 2. Install dependencies

```bash
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Add New Person to Database

Place folders of new individuals in `datasets/new_persons/`, each folder named after the person. Example:

```
datasets/new_persons/
├── Seong_Gi_Hun/
│   ├── 1.jpg
│   ├── 2.jpg
├── front_man/
│   ├── a.jpg
```

Then run:

```bash
python add_person.py \
  --backup-dir datasets/backup \
  --add-persons-dir datasets/new_persons \
  --faces-save-dir datasets/data \
  --features-path datasets/face_features/feature
```

### 2. Run Inference on Video

Place your video as `input_video.mp4` and run:

```bash
python main.py
```

This will generate an output video `output_video.mp4` with recognized face identities.

---

## 🧠 Model Details

### 🔍 Detection

* **Model**: YOLOv8n trained for facial detection.
* **Input**: Full video frame
* **Output**: Bounding boxes and facial landmarks

### 🧬 Recognition

* **Model**: ArcFace with IResNet100 backbone
* **Input**: Aligned and normalized face (112x112)
* **Output**: 512-D face embeddings
* **Matching**: Cosine similarity with existing embeddings in the `.npz` database

---
