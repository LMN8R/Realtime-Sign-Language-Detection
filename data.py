from function import *

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                frame = cv2.imread(f'Image/{action}/{sequence}.png')
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)

                cv2.putText(image, f'Extracting: {action} seq {sequence}', (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.imshow('Extracting Keypoints', image)
                cv2.waitKey(1)

                keypoints = extract_keypoints(results)
                np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)

    cv2.destroyAllWindows()
