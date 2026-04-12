from function import *
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = [(245,117,16), (117,245,16), (16,117,245), (245,16,117), (16,245,117)]

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    top5_idx = np.argsort(res)[::-1][:5]
    for i, idx in enumerate(top5_idx):
        color = colors[i] if res[idx] > threshold else (100,100,100)
        cv2.rectangle(output_frame, (0, 60+i*40), (int(res[idx]*100), 90+i*40), color, -1)
        cv2.putText(output_frame, f"{actions[idx]} {res[idx]*100:.0f}%", (0, 85+i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.6

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        ret, frame = cap.read()

        # Crop ROI and run detection
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0,40), (300,400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)

        # Draw hand skeleton on the cropped region
        draw_styled_landmarks(image, results)
        frame[40:400, 0:300] = image

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-15:]

        if len(sequence) == 15:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-5:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 10:
                sentence = sentence[-10:]

            # Confidence bars (top 5)
            frame = prob_viz(res, actions, frame, colors, threshold)

        # Output bar
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        display = ' '.join(sentence)
        if sentence and sentence[-1] != ' ':
            last_conf = res[np.argmax(res)]*100 if len(sequence) == 15 else 0
            display += f"  ({last_conf:.0f}%)"
        cv2.putText(frame, display, (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):       # Space: add space between letters
            sentence.append(' ')
        elif key == 8:              # Backspace: delete last letter
            if sentence:
                sentence.pop()

    cap.release()
    cv2.destroyAllWindows()
