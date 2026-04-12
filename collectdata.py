import os
import cv2

IMAGE_DIR = 'Image'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    roi = frame[40:400, 0:300].copy()

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    # Show image count per letter in the top-right corner
    for i, letter in enumerate(LETTERS):
        count = len(os.listdir(f'{IMAGE_DIR}/{letter}'))
        x = 310 + (i % 13) * 25
        y = 60 + (i // 13) * 20
        cv2.putText(frame, f'{letter}:{count}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 1)

    cv2.putText(frame, 'Press a letter key to save. ESC to quit.', (5, 30),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow('Collect Data', frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # Escape
        break
    if 97 <= key <= 122:  # a-z
        letter = chr(key).upper()
        count = len(os.listdir(f'{IMAGE_DIR}/{letter}'))
        cv2.imwrite(f'{IMAGE_DIR}/{letter}/{count}.png', roi)

cap.release()
cv2.destroyAllWindows()
