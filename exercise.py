import cv2 as cv
from poseModule import PoseDetector
import numpy as np
import math
import os
import argparse

detector = PoseDetector()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-e', '--exercise', default='push',  help='chose from push, pull and abdominal', type=str)
args = parser.parse_args()


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


# variables
count = 0
position = 'up'
angle, startAngle, endAngle =None, None, None
exercise = 'pull up'


def push_pull(pull=False):
    global count, position, angle, startAngle, endAngle, volBar, volPer
    start = 183
    end = 300
    if pull:
        start = 40
        end = 165

    startAngle, endAngle = start, end

    draw(points[11], points[13])
    draw(points[13], points[15])
    draw(points[12], points[14])
    draw(points[14], points[16])

    angle = calc_angle(np.array(points[11][1:]), np.array(points[13][1:]), np.array(points[15][1:]))

    if points[12][2] and points[11][2] >= points[14][2] and points[13][2]:
        position = 'down'

    if (points[12][2] and points[11][2] <= points[14][2] and points[13][2]) and position == 'down':
         count+=1
         position = 'up'


def abdominal():
    global count, position, angle, startAngle, endAngle, volBar, volPer
    start = 80
    end = 170
    startAngle, endAngle = start, end

    draw(points[11], points[23])
    draw(points[23], points[25])
    draw(points[12], points[24])
    draw(points[24], points[26])
    draw(points[23], points[24])

    angle = calc_angle(np.array(points[11][1:]), np.array(points[23][1:]), np.array(points[25][1:]))

    if points[25][2] and points[26][2] >= points[23][2] and points[24][2]:
        position = 'down'

    if (points[25][2] and points[26][2] <= points[23][2] and points[24][2]) and position == 'down':
         count+=1
         position = 'up'


cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'MP4V')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
name = 'data/video'+str(len(os.listdir('data'))+1)+'.mp4'
writer = cv.VideoWriter(name, fourcc, 30, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detector.estimate(frame, draw=False)
    points = detector.findPostions(frame, draw=False)
    if len(points) > 0:
        if args.exercise == 'push':
            push_pull()
            volBar = np.interp(angle, [startAngle, endAngle], [400, 150])
            volPer = np.interp(angle, [startAngle, endAngle], [0, 100])
        elif args.exercise == 'pull':
            push_pull(pull=True)
            volBar = np.interp(angle, [startAngle, endAngle], [150, 400])
            volPer = np.interp(angle, [startAngle, endAngle], [100, 0])
        elif args.exercise == 'abdominal':
            abdominal()
            volBar = np.interp(angle, [startAngle, endAngle], [150, 400])
            volPer = np.interp(angle, [startAngle, endAngle], [100, 0])

        # bar display
        cv.rectangle(frame, (50, 150), (60, 400), (230, 230, 230), cv.FILLED)
        cv.rectangle(frame, (50, int(volBar)), (60, 400), (30, 30, 30), cv.FILLED)
        cv.putText(frame, str(int(volPer)) + "%", (88, int(volBar)), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 2)

        # counter desplay
        cv.circle(frame, (60, 40), 33, (320, 320, 320), cv.FILLED)
        cv.circle(frame, (60, 40), 33, (50, 50, 50), 5)
        if count < 10:
            cv.putText(frame, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
            cv.putText(frame, str(count), (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    writer.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv.destroyAllWindows()
