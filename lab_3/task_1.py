import time
import cv2
from cv2.typing import MatLike

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')

REMAIN_VIDEO_SECONDS = 10


def recognize_frame(frame: MatLike) -> None:
    """
    Perform face and eye recognition on a frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Обличчя не знайдені на кадрі.")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = EYE_CASCADE.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)


def display_video_recognition(object_path: str, is_video: bool = True) -> None:
    """
    Process and display face and eye recognition on video or image.
    """
    if is_video:
        cap = cv2.VideoCapture(object_path)

        if not cap.isOpened():
            print("Не вдалося відкрити відео.")
            exit()

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        max_frames = REMAIN_VIDEO_SECONDS * fps
        frame_count = 0

        pause_pressed = False
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Відео завершено або не вдалося прочитати кадр.")
                break
            recognize_frame(frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                pause_pressed = True

            while pause_pressed:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pause_pressed = not pause_pressed

        cap.release()
        cv2.destroyAllWindows()

    else:
        image = cv2.imread(object_path)
        if image is None:
            print(f"Не вдалося відкрити зображення: {object_path}")
            return

        start = time.time()
        recognize_frame(image)

        while time.time() - start < REMAIN_VIDEO_SECONDS:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    objects = [
        # ('input/1.jpg', False),
        # ('input/2.jpg', False),
        # ('input/general.jpg', False),
        ('input/3444516-hd_1920_1080_30fps.mp4', True),
        ('input/face-demographics-walking.mp4', True),
        ('input/head-pose-face-detection-female.mp4', True),
        ('input/head-pose-face-detection-female-and-male.mp4', True),
        ('input/self.mp4', True),
    ]

    for object_path, is_video in objects:
        display_video_recognition(object_path, is_video)
