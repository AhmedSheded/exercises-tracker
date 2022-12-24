import cv2 as cv
import mediapipe as mp
mp_pose = mp.solutions.pose

class PoseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
               smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                      self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_tracking_confidence)


    def estimate(self, frame, draw=False):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.resluts = self.pose.process(frameRGB)

        if draw:
            self.drawing.draw_landmarks(frame, self.resluts.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                               landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style())
        return frame

    def findPostions(self, frame, draw=False):
        poseList = []
        if self.resluts.pose_landmarks:
            h, w = frame.shape[:2]
            for id, lm in enumerate(self.resluts.pose_landmarks.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                poseList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 15, (255, 255, 0), cv.FILLED)

        return poseList

def main():
    cap = cv.VideoCapture(0)
    detector = PoseDetector()

    while cap.isOpened():
        ret, frame = cap.read()

        frame = detector.estimate(frame, draw=True)
        postions = detector.findPostions(frame, draw=False)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()
