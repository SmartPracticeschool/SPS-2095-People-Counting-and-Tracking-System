# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:52:03 2020

@author: sai seravan
"""

import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from packages.centeroidTracking import CentroidTracker
from packages.trackableObject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from flask import Flask , render_template , Response
app = Flask(__name__)
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(r"modelFiles/MobileNetSSD_deploy.prototxt", r"modelFiles/MobileNetSSD_deploy.caffemodel")
# if a video path was not supplied, grab a reference to the webcam
if not (r"videosInput/inputclip2.mp4", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture("videosInput/inputclip2.mp4")
@app.route('/')
def index():
    return render_template('base.html')
def gen():
    # initialize the video writer (we'll instantiate later if need be)
    writer = None
    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    skip_frames = 30
    fps = FPS().start()
    while True:
        sucess ,frame = vs.read()
        if "videosInput/inputclip2.mp4" is not None and frame is None:
            break
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        if "output_footage" is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(r"output_footage/outputclip.avi", fourcc, 30,
			(W, H), True)
            status = "Waiting"
            rects = []
        if totalFrames % skip_frames == 0:
            status = "Detecting"
            trackers = []
            
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
			    # filter out weak detections by requiring a minimum
			    # confidence
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    
                    trackers.append(tracker)
        else:
            # loop over the trackers
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
        # draw a horizontal line in the center of the frame -- once an
	    # object crosses this line we will determine whether they were
	    # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)
	    # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True
				    # if the direction is positive (indicating the object
				    # is moving down) AND the centroid is below the
				    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imwrite("3.jpg",frame)
        # construct a tuple of information we will be displaying on the
	    # frame
        info = [
            ("Up", totalUp),
		    ("Down", totalDown),
		    ("Status", status),]
	    # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        (flag,encodedImage)=cv2.imencode(".jpg",frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype = 'multipart/x-mixed-replace;boundary=frame')
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=False)
    vs.release()
    cv2.destroyAllWindows()