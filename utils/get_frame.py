#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

def getFrames(feed):
    """Get frames from a live feed

    Arguments:
        feed {string} -- Path to the live feed.

    Returns:
        frame -- Feed frame
    """
    cap = cv2.VideoCapture(feed)
    for i in range(5):
        _,frame = cap.read()
        cv2.imwrite("static/images/feed/%d.jpg" % i, frame)
    cap.release()
    return frame

    

    

