import time

import cv2

cap = cv2.VideoCapture('data/eye2.mp4')

eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        gray_roi = gray[ey:ey + eh, ex:ex + ew]
        roi = frame[ey:ey + eh, ex:ex + ew]

        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        _, threshold = cv2.threshold(gray_roi, 10, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

            center_x = x + int(w / 2)
            center_y = y + int(h / 2)
            rows, cols = roi.shape[:2]
            cv2.line(roi, (center_x, 0), (center_x, rows), (0, 255, 0), 2)  # Vertical line
            cv2.line(roi, (0, center_y), (cols, center_y), (0, 255, 0), 2)  # Horizontal line

            break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('here!')
        while cv2.waitKey(1) & 0xFF != ord('q'):
            time.sleep(1)

cap.release()
cv2.destroyAllWindows()
