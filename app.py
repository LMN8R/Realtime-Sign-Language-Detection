from function import *
from keras.models import model_from_json

json_file = open("model.json", "r")
model = model_from_json(json_file.read())
json_file.close()
model.load_weights("model.h5")

colors = [(52, 235, 131), (52, 174, 235), (235, 52, 174), (235, 174, 52), (174, 52, 235)]


def draw_confidence_panel(frame, res, actions, threshold):
    top5_idx = np.argsort(res)[::-1][:5]
    panel_x = 315

    cv2.rectangle(frame, (panel_x, 50), (640, 265), (30, 30, 30), -1)
    cv2.putText(frame, 'Top predictions', (panel_x + 6, 70),
                cv2.FONT_HERSHEY_PLAIN, 1, (180, 180, 180), 1)

    for i, idx in enumerate(top5_idx):
        y = 82 + i * 38
        bar_len = int(res[idx] * (640 - panel_x - 12))
        color = colors[i] if res[idx] > threshold else (70, 70, 70)
        cv2.rectangle(frame, (panel_x + 6, y), (panel_x + 6 + bar_len, y + 22), color, -1)
        cv2.putText(frame, f'{actions[idx]}  {res[idx]*100:.0f}%', (panel_x + 9, y + 16),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)


sequence = []
sentence = []
predictions = []
threshold = 0.6
res = np.zeros(len(actions))

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        image, results = mediapipe_detection(cropframe, hands)
        draw_styled_landmarks(image, results)
        frame[40:400, 0:300] = image

        hand_detected = results.multi_hand_landmarks is not None

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-15:]

        if len(sequence) == 15:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-5:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if not sentence or actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 10:
                sentence = sentence[-10:]

            draw_confidence_panel(frame, res, actions, threshold)

        # Hand status + frame buffer progress bar
        status_y = 280
        if hand_detected:
            cv2.putText(frame, 'Hand detected', (318, status_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (52, 235, 131), 1)
            progress = int((len(sequence) / 15) * 322)
            cv2.rectangle(frame, (315, status_y + 8), (637, status_y + 22), (50, 50, 50), -1)
            cv2.rectangle(frame, (315, status_y + 8), (315 + progress, status_y + 22), (52, 235, 131), -1)
        else:
            cv2.putText(frame, 'No hand in frame', (318, status_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (80, 80, 200), 1)

        # Controls hint at the bottom
        cv2.putText(frame, 'Space: space   Bksp: delete   C: clear   Q: quit',
                    (5, frame.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN, 0.9, (160, 160, 160), 1)

        # Output bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
        display = ' '.join(sentence)
        if sentence and sentence[-1] != ' ':
            display += f'  ({res[np.argmax(res)]*100:.0f}%)'
        cv2.putText(frame, display, (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detection', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            sentence.append(' ')
        elif key == ord('c'):
            sentence.clear()
            predictions.clear()
        elif key == 8:
            if sentence:
                sentence.pop()

    cap.release()
    cv2.destroyAllWindows()
