# Real-Time ASL Alphabet Detection

This project recognizes American Sign Language alphabet letters (`A-Z`) from a webcam feed in real time.

The core idea is simple: instead of training directly on raw images, it first uses MediaPipe Hands to detect 21 hand landmarks, normalizes those landmarks so hand position and scale matter less, and then feeds short landmark sequences into an LSTM classifier.

That design keeps the project lightweight enough for fast training and real-time inference while still handling short temporal patterns.

## What the app does

- Opens the webcam and asks the user to place their hand inside a fixed region of interest.
- Runs MediaPipe hand tracking on that cropped region.
- Converts the detected landmarks into a 63-value feature vector (`21 landmarks x 3 coordinates`).
- Maintains a rolling 5-frame sequence of those features.
- Runs the trained model on the sequence and shows the top predictions with confidence bars.
- Only commits a letter to the output when the prediction is both confident and stable across recent frames.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Real-time inference app |
| `collectdata.py` | Webcam image collection script |
| `data.py` | Converts saved images into MediaPipe landmark arrays |
| `trainmodel.py` | Loads landmark sequences, trains the model, and saves evaluation artifacts |
| `function.py` | Shared constants and utility functions |
| `model.json` / `model.h5` | Saved model architecture and weights |
| `Image/` | Optional webcam captures from the original prototype |
| `MP_Data/` | Landmark arrays grouped by letter and sequence |
| `Logs/` | TensorBoard logs and saved evaluation output |

## Pipeline

### 1. Data collection

`collectdata.py` captures cropped hand images from the webcam and stores them under `Image/<LETTER>/`. It is still useful for collecting quick webcam-specific examples, but the current training pipeline is built around the Kaggle dataset in `asl_dataset/`.

Example:

```bash
python collectdata.py
```

Press a letter key to save the current hand crop into that letter's folder. Press `Esc` to quit.

### 2. Landmark extraction

`data.py` runs every saved image through MediaPipe and stores normalized landmark vectors in `MP_Data/`.

Example:

```bash
python data.py
```

Each saved `.npy` file contains 63 features:

- `x`, `y`, and `z` for each of the 21 hand landmarks
- translated so the wrist is the origin
- scaled relative to hand size

### 3. Model training

`trainmodel.py` loads the landmark sequences, splits them into train/test sets, trains the LSTM model, evaluates it, and saves:

- `model.json`
- `model.h5`
- `Logs/evaluation_report.txt`
- `Logs/confusion_matrix.csv`

Example:

```bash
python trainmodel.py
```

### 4. Live inference

`app.py` loads the trained model and runs real-time prediction from webcam input.

Example:

```bash
python app.py
```

Controls:

- `Q` quit
- `Space` insert a space in the output
- `Backspace` delete the last item
- `C` clear the current output

## Model architecture

Input shape: `5 x 63`

```text
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64)
Dense(64)
Dropout(0.3)
Dense(32)
Dense(26, softmax)
```

Why this setup:

- MediaPipe handles the heavy computer vision work.
- Landmark sequences are much smaller than raw images.
- LSTMs are a reasonable fit for short temporal patterns and help smooth noisy frame-by-frame predictions.

## Design decisions

### Why landmarks instead of raw images?

Training directly on pixels would usually require much more data and compute. Using landmarks makes the model focus on hand geometry rather than background clutter, lighting, or camera details.

### Why normalize the landmarks?

The landmark coordinates are shifted so the wrist becomes the origin, then scaled by hand size. That reduces sensitivity to where the hand appears in the crop and how close it is to the camera.

### Why keep a rolling sequence?

A single frame can be noisy. A short sequence makes the prediction more stable and reduces flicker during live use.

### Why add confidence and stability checks?

For a webcam demo, wrong confident outputs are more distracting than delayed outputs. Requiring a stable prediction across recent frames makes the interaction feel much more reliable.

## Current limitations

This project works as a personal demo, but there are some important caveats:

- The dataset is small and was collected manually, so generalization is limited.
- The training data is not session-diverse or signer-diverse.
- Some letters are naturally harder to separate because their hand shapes are very similar.
- The train/test split is performed at the sequence level, so the reported score should be treated as an internal benchmark rather than a final real-world accuracy claim.
- The current model is trained on grouped 5-frame image sequences from the Kaggle dataset, which is much better than duplicated still frames, but it still does not perfectly match a live webcam environment.

If I continued the project, the next steps would be to fine-tune on webcam-specific sequences, test on more users, and expand the training set for letters that are visually similar or motion-dependent.

## Setup

Python `3.9` is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```
