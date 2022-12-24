import cv2 as cv
from poseModule import PoseDetector
import numpy as np
import math

detector = PoseDetector()


def pixelInCm(x1, y1, x2, y2):
    dx = x1-x2
    dy = y1-y2
    pixels = np.sqrt(dx*dx+dy*dy)
    pixel = pixels/30
    return pixel


def calc_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang



def draw(point1, point2):
    x1, y1 = point1[1], point1[2]
    x2, y2 = point2[1], point2[2]
    cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv.circle(frame, (x2, y2), 5, (255, 255, 255), cv.FILLED)
    cv.circle(frame, (x1, y1), 5, (255, 255, 255), cv.FILLED)
    cv.circle(frame, (x1, y1), 10, (230, 230, 230), 5)
    cv.circle(frame, (x2, y2), 10, (230, 230, 230), 5)

count = 0
position = None


cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    frame = detector.estimate(frame, draw=False)
    points = detector.findPostions(frame, draw=False)
    if len(points)>0:
        pixel = pixelInCm(points[11][1], points[11][2], points[12][1], points[12][2])
        x1, y1 = points[11][1], points[11][2]
        x2, y2 = points[15][1], points[15][2]
        length = math.hypot(x2 - x1, y2 - y1) // pixel
        destance = (y2-y1)//pixel
        # cv.putText(frame, str(destance)+'  cm', (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        draw(points[11], points[13])
        draw(points[13], points[15])
        draw(points[12], points[14])
        draw(points[14], points[16])


        angle = calc_angle(np.array(points[11][1:]), np.array(points[15][1:]), np.array(points[12][1:]))


        if points[12][2] and points[11][2] >= points[14][2] and points[13][2]:
            position='down'

        if (points[12][2] and points[11][2] <= points[14][2] and points[13][2]) and position == 'down':
            count+=1
            position = 'up'
            print(position)

        volBar = np.interp(length, [0, 14], [400, 150])
        volPer = np.interp(length, [0, 14], [-7, 100])
        cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 255), 3)
        cv.rectangle(frame, (50, int(volBar)), (85, 400), (200, 150, 200), cv.FILLED)
        cv.putText(frame, str(int(volPer)) + "%", (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 150, 255), 2)

        cv.putText(frame, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)



    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break
