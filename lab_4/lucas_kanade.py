import cv2
import numpy as np


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    old_frame = cv2.resize(old_frame, (800, 600))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color=(0, 0, 255), thickness=-1)

            img = cv2.add(frame, mask)

            cv2.imshow('Ball Tracking', img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                while True:
                    pass

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


# process_video('videos/ball.mp4')
process_video(0)
