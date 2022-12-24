import cv2 as cv
import mediapipe as mp

drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_styles = mp.solutions.drawing_styles

cap = cv.VideoCapture(0)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

forcc = cv.VideoWriter_fourcc(*'mp4v')
writer = cv.VideoWriter('pose estmation.mp4', forcc, 30, (height, width))
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # make detection
        resluts = pose.process(rgbFrame)

        drawing.draw_landmarks(frame, resluts.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style())

        writer.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(30) == 27:
            break
    else:
        break
cap.release()
cv.destroyAllWindows()