from function import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau


def augment_sequence(window):
    sequence = np.array(window, dtype=np.float32)
    augmented = []

    base_scale = np.random.uniform(0.98, 1.02)
    drift = np.random.uniform(-0.01, 0.01, size=(3,))
    per_frame_noise = np.random.normal(0, 0.005, size=sequence.shape)

    for frame_idx, frame in enumerate(sequence):
        landmarks = frame.reshape(21, 3).copy()
        progress = frame_idx / max(len(sequence) - 1, 1)

        landmarks *= base_scale
        landmarks += drift * progress
        landmarks += per_frame_noise[frame_idx].reshape(21, 3)

        augmented.append(landmarks.flatten())

    return np.array(augmented, dtype=np.float32)


def load_sequences():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    sequence_counts = {}

    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_dir):
            continue

        sequence_dirs = sorted(
            [path for path in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, path))],
            key=lambda name: int(name)
        )
        sequence_counts[action] = len(sequence_dirs)

        for sequence_name in sequence_dirs:
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_dir, sequence_name, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    window = []
                    break
                window.append(np.load(frame_path))

            if len(window) != sequence_length:
                continue

            sequences.append(window)
            labels.append(label_map[action])
            sequences.append(augment_sequence(window))
            labels.append(label_map[action])

    return np.array(sequences), np.array(labels), sequence_counts


def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def main():
    X, labels_array, sequence_counts = load_sequences()
    y = to_categorical(labels_array).astype(int)

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X,
        y,
        labels_array,
        test_size=0.2,
        random_state=42,
        stratify=labels_array
    )

    log_dir = os.path.join('Logs')
    os.makedirs(log_dir, exist_ok=True)
    callbacks = [
        TensorBoard(log_dir=log_dir),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
    ]

    model = build_model()
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=callbacks)
    model.summary()

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy*100:.1f}%")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(labels_test, y_pred, target_names=actions, digits=3, zero_division=0)
    matrix = confusion_matrix(labels_test, y_pred)
    per_class_accuracy = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)

    print("\nSequences per label:\n")
    for action in actions:
        print(f"{action}: {sequence_counts.get(action, 0)}")

    print("\nClassification report:\n")
    print(report)

    report_path = os.path.join(log_dir, "evaluation_report.txt")
    with open(report_path, "w") as report_file:
        report_file.write("Sequences per label:\n")
        for action in actions:
            report_file.write(f"{action}: {sequence_counts.get(action, 0)}\n")
        report_file.write(f"\nTest accuracy: {accuracy*100:.1f}%\n\n")
        report_file.write("Per-class accuracy:\n")
        for action, class_acc in zip(actions, per_class_accuracy):
            report_file.write(f"{action}: {class_acc*100:.1f}%\n")
        report_file.write("\nClassification report:\n")
        report_file.write(report)

    matrix_path = os.path.join(log_dir, "confusion_matrix.csv")
    header = ",".join(["label"] + actions.tolist())
    rows = [header]
    for action, row in zip(actions, matrix):
        rows.append(",".join([action] + [str(value) for value in row]))

    with open(matrix_path, "w") as matrix_file:
        matrix_file.write("\n".join(rows))

    print(f"Saved evaluation report to {report_path}")
    print(f"Saved confusion matrix to {matrix_path}")

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save('model.h5')
    print("Model saved.")


if __name__ == "__main__":
    main()
