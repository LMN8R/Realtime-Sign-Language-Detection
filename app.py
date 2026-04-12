from function import *
from keras.models import model_from_json

colors = [(52, 235, 131), (52, 174, 235), (235, 52, 174), (235, 174, 52), (174, 52, 235)]
SMOOTHING_WINDOW = 5
STABILITY_WINDOW = 5
MIN_CONFIDENCE_MARGIN = 0.15
ROI_TOP = 40
ROI_BOTTOM = 400
ROI_LEFT = 0
ROI_RIGHT = 300
LETTER_MARGIN_OVERRIDES = {
    'I': 0.05,
    'J': 0.05,
    'M': 0.08,
    'N': 0.08,
    'O': 0.08,
}


def required_margin(label):
    return LETTER_MARGIN_OVERRIDES.get(label, MIN_CONFIDENCE_MARGIN)


def resolve_dynamic_letters(sequence, res, actions):
    top_candidates = np.argsort(res)[::-1][:3]
    candidate_labels = {actions[idx] for idx in top_candidates}

    if not {'I', 'J'} & candidate_labels:
        return int(np.argmax(res))

    pinky_positions = np.array([frame[20 * 3:20 * 3 + 2] for frame in sequence])
    path_length = np.sum(np.linalg.norm(np.diff(pinky_positions, axis=0), axis=1))
    displacement = np.linalg.norm(pinky_positions[-1] - pinky_positions[0])

    if path_length > 0.12 or displacement > 0.07:
        return int(np.where(actions == 'J')[0][0])

    return int(np.where(actions == 'I')[0][0])


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


def load_trained_model():
    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("model.h5")
    return model


def main():
    model = load_trained_model()
    sequence = []
    sentence = []
    predictions = []
    prediction_history = []
    threshold = 0.6
    res = np.zeros(len(actions))
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cropframe = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
            frame = cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 255, 255), 2)
            image, results = mediapipe_detection(cropframe, hands)
            draw_styled_landmarks(image, results)
            frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = image

            hand_detected = results.multi_hand_landmarks is not None
            confidence_margin = 0.0
            margin_threshold = MIN_CONFIDENCE_MARGIN

            if hand_detected:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
            else:
                sequence.clear()
                predictions.clear()
                prediction_history.clear()
                res = np.zeros(len(actions))

            if len(sequence) == sequence_length:
                raw_res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction_history.append(raw_res)
                prediction_history[:] = prediction_history[-SMOOTHING_WINDOW:]
                res = np.mean(prediction_history, axis=0)
                predicted_idx = resolve_dynamic_letters(sequence, res, actions)
                sorted_scores = np.sort(res)[::-1]
                confidence_margin = sorted_scores[0] - sorted_scores[1]
                predictions.append(predicted_idx)
                predictions = predictions[-STABILITY_WINDOW:]
                predicted_label = actions[predicted_idx]
                margin_threshold = required_margin(predicted_label)

                stable_prediction = (
                    len(predictions) == STABILITY_WINDOW and
                    len(set(predictions)) == 1
                )

                if stable_prediction and res[predicted_idx] > threshold and confidence_margin > margin_threshold:
                    if not sentence or predicted_label != sentence[-1]:
                        sentence.append(predicted_label)

                if len(sentence) > 10:
                    sentence = sentence[-10:]

                draw_confidence_panel(frame, res, actions, threshold)

            status_y = 280
            if hand_detected:
                cv2.putText(frame, 'Hand detected', (318, status_y),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, (52, 235, 131), 1)
                progress = int((len(sequence) / sequence_length) * 322)
                cv2.rectangle(frame, (315, status_y + 8), (637, status_y + 22), (50, 50, 50), -1)
                cv2.rectangle(frame, (315, status_y + 8), (315 + progress, status_y + 22), (52, 235, 131), -1)
                if len(sequence) == sequence_length and res[np.argmax(res)] <= threshold:
                    cv2.putText(frame, 'Prediction below confidence threshold', (318, status_y + 38),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 180, 240), 1)
                elif len(sequence) == sequence_length and confidence_margin <= margin_threshold:
                    cv2.putText(frame, 'Prediction ambiguous, hold the sign steady', (318, status_y + 38),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 180, 240), 1)
            else:
                cv2.putText(frame, 'No hand in frame', (318, status_y),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, (80, 80, 200), 1)

            cv2.putText(frame, 'Space: space   Bksp: delete   C: clear   Q: quit',
                        (5, frame.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN, 0.9, (160, 160, 160), 1)

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
                prediction_history.clear()
                sequence.clear()
                res = np.zeros(len(actions))
            elif key == 8 and sentence:
                sentence.pop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
