from function import *
import shutil

SOURCE_DATASET = 'asl_dataset'


def dataset_root():
    nested_root = os.path.join(SOURCE_DATASET, SOURCE_DATASET)
    if os.path.isdir(nested_root):
        return nested_root
    return SOURCE_DATASET


def group_sequence_files(label_dir):
    grouped = {}

    for file_name in os.listdir(label_dir):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        parts = file_name.split('_')
        if len(parts) < 5 or not parts[3] == 'seg':
            continue

        group_key = "_".join(parts[:3])
        frame_idx = int(parts[4]) - 1
        grouped.setdefault(group_key, {})[frame_idx] = os.path.join(label_dir, file_name)

    complete_groups = []
    for group_key in sorted(grouped.keys()):
        frames = grouped[group_key]
        if len(frames) != sequence_length:
            continue
        complete_groups.append([frames[idx] for idx in range(sequence_length)])

    return complete_groups


def rebuild_output_directory():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    for action in actions:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)


def main():
    rebuild_output_directory()
    root_dir = dataset_root()

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for action in actions:
            label_dir = os.path.join(root_dir, action.lower())
            grouped_sequences = group_sequence_files(label_dir)

            for sequence_idx, frame_paths in enumerate(grouped_sequences):
                sequence_dir = os.path.join(DATA_PATH, action, str(sequence_idx))
                os.makedirs(sequence_dir, exist_ok=True)

                for frame_num, frame_path in enumerate(frame_paths):
                    frame = cv2.imread(frame_path)
                    image, results = mediapipe_detection(frame, hands)
                    draw_styled_landmarks(image, results)

                    cv2.putText(image, f'Extracting: {action} seq {sequence_idx}', (10, 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.imshow('Extracting Keypoints', image)
                    cv2.waitKey(1)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(sequence_dir, str(frame_num)), keypoints)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
