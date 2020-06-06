import numpy as np
import cv2
import imutils
import logging
import argparse
from collections import deque
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils.trajectory_functions import quadratic_eqn
from utils.yolo import initialise_yolo

# initialise logging
logging.basicConfig(filename='logging.log', format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG)

# initialise argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video")
args = vars(ap.parse_args())
# going to initialise some variables we need for logic
writer = None  # will be used to write the output to disk
trace_location = None  # stores the location a line should be drawn in a given frame
ball_found = False  # determines if the ball has been located within a frame
object_lost_count = 0  # counter to track the number of consecuative frames the ball has been lost by the YOLO model
tracked_points = deque(maxlen=128)  # list-like structure we will use to store the ball locations from each frame
ball_found_initially = None  # flags when we have found the ball for the first time
frame_number = 0  # tracks frame number (for logging)
prediction_count = 0  # tracks number of predictions made
confidence_threshold = 0.5  # confidence threshold for yolo model to detect object
suppression_threshold = 0.3  # will be used when we apply non-maxima suppression to identify balls bounding box


input_video = args["video"]


output_file_name = input_video.split('/')[-1]
output_location = f"data/output_data/{output_file_name}"

# initialise the model
logging.info("Initialising YOO model")
yolo_model, ln, labels = initialise_yolo()

# initialise the video stream
logging.info(f"Creating video stream object from file at {input_video}")
vs = cv2.VideoCapture(input_video)

# loop over frames from the video stream
logging.info("Iterating through frames")
while True:
    # grab the next frame in the video stream
    (grabbed, frame) = vs.read()
    ball_found_in_frame = False

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # use the yolo model to detect objects in image and get their bounding boxes
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    layerOutputs = yolo_model.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # get the class id and confidence for the object
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:

                # get the bounding box for the object
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to refine the balls bounding box
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, suppression_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # we only care about the ball
            if labels[classIDs[i]] == "sports ball":

                # let the system know we found the ball
                ball_found_initially = True
                ball_found_in_frame = True
                object_lost_count = 0
                prediction_count = 0

                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                trace_location = (int(x + (w / 2)), int(y + (h / 2)))

    # if we couldn't find the ball, increase the counter
    if not ball_found_in_frame:
        object_lost_count += 1

    # if we have lost the ball for 10 frames, lets get the trajectory so we can start making predictions
    if ball_found_initially and (object_lost_count == 10) and (len(tracked_points) > 100):
        logging.info(f"ball has be lost at frame number {frame_number}, fitting curve to approximate trajectory")
        x_train = [pt[0] for pt in tracked_points]
        y_train = [pt[1] for pt in tracked_points]
        x_train.reverse()
        y_train.reverse()

        # note we only use the last 20 points to fit the curve so we can get a better fit
        popt, pcov = curve_fit(quadratic_eqn, np.array(x_train[-20:]), np.array(y_train[-20:]))

    # basically if we have lost the ball and it is not too early or late in the video, let's make a prediction
    if ball_found_initially and (object_lost_count >= 10) and (len(tracked_points) > 100) and (prediction_count <= 10):
        x_train = [pt[0] for pt in tracked_points]
        x_train.reverse()

        # fit a simple linear regression model to predict the next x position
        x_train = x_train[-20:]
        times = [i for i in range(len(x_train))]

        reg = LinearRegression().fit(np.array(times).reshape(-1, 1), np.array(x_train).reshape(-1, 1))
        x = reg.predict(np.array([20]).reshape(1, -1))[0]

        # use our curve fit to predict the next y
        y = quadratic_eqn(x, popt[0], popt[1], popt[2])

        trace_location = (int(x + (w / 2)), int(y + (h / 2)))

        prediction_count += 1

    if trace_location:
        tracked_points.appendleft(trace_location)

        # loop over the set of tracked points
        for i in range(1, len(tracked_points)):
            # if either of the tracked points are None, ignore them
            if tracked_points[i - 1] is None or tracked_points[i] is None:
                continue
            # otherwise, draw a line connecting them
            cv2.line(frame, tracked_points[i - 1], tracked_points[i], (255, 0, 0), 2)

    # write the frame to our output file
    if writer is None:
        # initialize our video writer
        writer = cv2.VideoWriter(output_location,
                                 cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                 30, (frame.shape[1],
                                      frame.shape[0]),
                                 True)

    # write the output frame to disk
    writer.write(frame)
    frame_number += 1

logging.info("Finalising")
writer.release()
vs.release()
logging.info(f"Process complete, output has been saved at {output_location}")