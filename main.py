"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import math
import numpy as np



# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

def drawOutput(result, frame, originalWidth, originalHeight, tempCount,tempCheck, prob_threshold ):
    currentCount = 0
    length = tempCount

    for obj in result[0][0]:

        if obj[2] > prob_threshold:
            xmin = int(obj[3] * originalWidth)
            ymin = int(obj[4] * originalHeight)
            xmax = int(obj[5] * originalWidth)
            ymax = int(obj[6] * originalHeight)

            currentCount +=1
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            x = int((xmin + xmax) / 2)
            y = int((ymax + ymin) / 2)
            if x != 0 and y != 0:
                cv2.putText(frame,"Person Detected",(100, 50),cv2.FONT_HERSHEY_COMPLEX,0.5,(100, 100, 255),1)
                # to draw an arrow and see tat he is in frame
                cv2.arrowedLine(frame, (160, 50), (x, y), (255, 0, 255), 3)
                c_x = frame.shape[1] / 2
                c_y = frame.shape[0] / 2
                mid_x = (160 + x) / 2
                mid_y = (50 + y) / 2

                # Calculating the length of arrow
                length = math.sqrt(math.pow(mid_x - c_x, 2) + math.pow(mid_y - c_y, 2) * 1.0)
                tempCheck = 0

    if currentCount < 1:
        tempCheck +=1
    
    if length < 100 and tempCheck < 10:
        currentCount = 1
        tempCheck += 1
        if tempCheck > 100:
            tempCheck =0
    return frame, currentCount, length, tempCheck
                
def ssd_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    currentCount = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * originalWidth)
            ymin = int(obj[4] * originalHeight)
            xmax = int(obj[5] * originalWidth)
            ymax = int(obj[6] * originalHeight)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            currentCount = currentCount + 1
    return frame, currentCount


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.device,args.model)
    # Set Probability threshold for detections
    global prob_threshold,originalHeight,originalWidth
    prob_threshold = args.prob_threshold
    infer_network.load_model()

    imageFlag = False

    if args.input.lower()=="cam":
        videoPath = 0
    elif args.input.split(".")[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']:
        imageFlag = True
        videoPath =args.input
    else:
        if not os.path.isfile(args.input):
            # print(" Given input file is not present.")
            exit(1)
        videoPath = args.input
    stream = cv2.VideoCapture(videoPath)
    stream.open(videoPath)

    originalWidth = int(stream.get(3))
    originalHeight = int(stream.get(4))

    # n is batchsize, c is channels, h &w are height and width
    n, c, h ,w = infer_network.get_input_shape()

    lastCount = 0
    totalCount =0
    while stream.isOpened():
        flag , frame = stream.read()
        if not flag:
            break
        keyPress = cv2.waitKey(60)

        transformedImage = cv2.resize(frame, (w,h))
        transformedImage = transformedImage.transpose((2,0,1))

        #hardcoded c value to handle png images
        transformedImage = transformedImage.reshape(n,3,h,w)

        peopleCount = 0

        #start async req
        startTime = time.time()
        infer_network.exec_net(transformedImage)

        color = (255,0,0)

        if infer_network.wait() == 0:
            inferTime = time.time() - startTime
            result = infer_network.get_output()
            # print(result)
        
        #Drawing bouding boxes
        # frame, currentCount, length, tempCheck = drawOutput(result, frame, originalWidth, originalHeight, tempCount,tempCheck, args.prob_threshold )
        frame, currentCount = ssd_out(frame,result)
        #coverting to millli seconds
        inferTimeMessage = " No of persons: {:d}, Inference time: {:.3f}ms,".format(currentCount,inferTime*1000)

        cv2.putText(frame,inferTimeMessage,(15,15),cv2.FONT_HERSHEY_COMPLEX,0.5,color,1)

        if currentCount > lastCount:  # New entry
            start_time = time.time()
            totalCount = totalCount + currentCount - lastCount
            client.publish("person", json.dumps({"total": totalCount}))
        #Average time
        if currentCount < lastCount:  
            duration = int(time.time() - start_time)
            client.publish("person/duration", json.dumps({"duration": duration}))

        client.publish("person", json.dumps({"count": currentCount}))
        lastCount = currentCount

        cv2.imshow('Frame',frame)

        # sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if keyPress == 27:
            break

        if imageFlag:
            cv2.imwrite("output_image.jpg", frame)

    stream.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
