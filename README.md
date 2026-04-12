# Realtime Sign Language Detection

Detects ASL alphabet letters (A–Z) in real time from your webcam. Trained on my own hand data, the model hits **98.7% test accuracy**.

## How it works

Instead of feeding raw video frames into a CNN, the pipeline first runs MediaPipe's hand tracking model to extract 21 3D landmarks per frame. Those keypoints get normalized so that hand position and size don't affect predictions — the wrist becomes the origin and everything scales relative to palm size. A 15-frame buffer of these keypoints then goes into an LSTM network, which outputs a letter once it's confident across 5 consecutive predictions.

Splitting the problem into hand detection → sequence classification keeps the dataset small and training fast, without sacrificing accuracy.

## Setup

Requires Python 3.9.

```bash
pip install -r requirements.txt
python app.py
```

## Controls

| Key | Action |
|---|---|
| Hold hand in the white rectangle | Detect letter |
| `Space` | Add a space |
| `Backspace` | Delete last letter |
| `Q` | Quit |

The top-left bar shows the current output. The sidebar shows confidence for the top 5 predicted letters.

## Retraining on your own data

**1. Collect images**

Run `collectdata.py` and press a letter key to capture a frame for that letter. Aim for 30+ images per letter across different positions and lighting.

```bash
python collectdata.py
```

**2. Extract keypoints**

Runs each image through MediaPipe and saves normalized keypoint arrays to `MP_Data/`.

```bash
python data.py
```

**3. Train**

```bash
python trainmodel.py
```

Trains for up to 200 epochs. Early stopping kicks in once validation loss stops improving — usually around epoch 70–90.

## Architecture

```
Input: 15 frames × 63 features (21 landmarks × x/y/z)
  → LSTM(64)  + Dropout(0.2)
  → LSTM(128) + Dropout(0.2)
  → LSTM(64)
  → Dense(64) + Dropout(0.3)
  → Dense(32)
  → Dense(26, softmax)
```

The 63 features are the x/y/z coordinates for each hand landmark, normalized to be wrist-relative and scale-invariant. Using sequences of frames rather than single frames makes the classifier more stable — a letter only registers once the model agrees with itself across multiple frames.

## File overview

| File | Purpose |
|---|---|
| `app.py` | Real-time detection |
| `collectdata.py` | Collect training images from webcam |
| `data.py` | Extract keypoints from collected images |
| `trainmodel.py` | Train the LSTM classifier |
| `function.py` | Shared utilities — MediaPipe setup, keypoint extraction |
