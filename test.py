import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model

protoFile = "openpose/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "openpose/pose/coco/pose_iter_440000.caffemodel"
modelFile = "models/mymodel.h5"
input_source = "eat_129.mp4"
output_fileName = "eat_129.avi"
nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

inWidth = 368
inHeight = 368
threshold = 0.2

LABELS = [
    "Eat",
    "Sit",
    "Sleep",
    "Stand",
    "Walk"
]
n_steps = 24  # 32 timesteps per series

# Load the openpose model
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Load the action-recognition trained model
model = load_model(modelFile)

cap = cv2.VideoCapture(input_source)
success, frame = cap.read()
vid_writer = cv2.VideoWriter(output_fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                             (frame.shape[1], frame.shape[0]))

iterator = 0
openpose_output = []  # Will store the openpose time series data for recent n_steps
inteval = 1
sequence_start = 0  # starting location of circular array

while (cap.isOpened()):
    t = time.time()
    success, frame = cap.read()
    if success:
        frameCopy = np.copy(frame)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append((0, 0))
        if len(openpose_output) < n_steps and len(points) > 0:
            openpose_output.append(points)
            iterator = iterator + 1
        elif len(points) > 0:
            openpose_output[iterator % n_steps] = points
            iterator = iterator + 1
            X_ = np.asarray(openpose_output, dtype=np.float32)
            X_ = X_[np.newaxis, :, :]
            X_ = X_.reshape(X_.shape[0], X_.shape[1], X_.shape[2] * X_.shape[3])
            # result_prob = model.predict(X_, None)
            sequence_arr = np.append(X_[:, sequence_start:, :], X_[:, :sequence_start, :], axis=1)
            sequence_start = (sequence_start + 1) % n_steps

            result_prob = model.predict(sequence_arr)
            y_class = result_prob.argmax(axis=-1)
            label = LABELS[y_class[0]]
            # label = "Walk"
            print("Iterator::", iterator, " Label ::", label)
            cv2.putText(frame, label, (50, 150), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), lineType=cv2.LINE_AA)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA][0] and points[partB][0]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)

        plt.figure(figsize=[10, 10])
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow('test', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # print(iterator)
        vid_writer.write(frame)

        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break
    else:
        break

vid_writer.release()
cap.release()
cv2.destroyAllWindows()
