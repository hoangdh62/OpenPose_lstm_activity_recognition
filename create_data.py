import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from imutils import paths

protoFile = "openpose/pose/COCO/pose_deploy_linevec.prototxt"
weightsFile = "openpose/pose/COCO/pose_iter_440000.caffemodel"
video_train_paths = list(paths.list_files('mydata/videos_train_Copy/Walk'))
nPoints = 18

fileX = open("mydata/X_train_walk.txt", "a", encoding="utf-8")
fileY = open("mydata/Y_train_walk.txt", "a", encoding="utf-8")

inWidth = 368
inHeight = 368
threshold = 0.2

n_steps = 24  # 32 timesteps per series

# Load the openpose model
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Load the action-recognition trained model
# model = load_model(modelFile)
num_path=1
for videoPath in video_train_paths:
    print(num_path)
    # label = videoPath.split(os.path.sep)[-2]
    label = "Walk"
    if label =="Eat": n=1
    elif label == "Sit": n=2
    elif label == "Sleep":n=3
    elif label == "Stand":n=4
    elif label == "Walk":n=5

    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()
    # vid_writer = cv2.VideoWriter(output_fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
    #                              (frame.shape[1], frame.shape[0]))

    iterator = 0
    openpose_output = []  # Will store the openpose time series data for recent n_steps
    inteval = 1
    # sequence_start = 0  # starting location of circular array

    while (cap.isOpened()and iterator<44):
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
            points = ""

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > threshold:
                    # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    #             lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points = points + str(int(x)) + "," + str(int(y)) + ","
                else:
                    points = points + "0,0,"
            points = points.rstrip(",")
            if len(openpose_output) < 24 and len(points) > 0:
                openpose_output.append(points)
                iterator = iterator + 1
            elif len(points) > 0:
                for j in range(len(openpose_output)):
                    fileX.write(openpose_output[j]+"\n")
                fileY.write(str(n)+"\n")
                openpose_output.append(points)
                openpose_output.pop(0)
                iterator = iterator + 1
                # X_ = np.asarray(openpose_output, dtype=np.float32)
                # X_ = X_[np.newaxis, :, :]
                # X_ = X_.reshape(X_.shape[0], X_.shape[1], X_.shape[2] * X_.shape[3])
                # # result_prob = model.predict(X_, None)
                # sequence_arr = np.append(X_[:, sequence_start:, :], X_[:, :sequence_start, :], axis=1)
                # sequence_start = (sequence_start + 1) % 32
            print(iterator)
            c = cv2.waitKey(inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break
        else:
            break
    num_path+=1
    cap.release()
    cv2.destroyAllWindows()
fileX.close()
fileY.close()


