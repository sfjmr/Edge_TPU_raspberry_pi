#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection Raspberry Pi camera stream.
    Copyright (c) 2019 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import io
import time

import numpy as np
import picamera
from picamera.array import PiRGBArray

import edgetpu.detection.engine
#from edgetpu.detection.engine import DetectionEngine

import cv2
import PIL

from utils import visualization as visual

WINDOW_NAME = 'Edge TPU Tf-lite object detection'

# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model', help='File path of Tflite model.', default='mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')#required=True, 
    parser.add_argument(
            '--label', help='File path of label file.', default='coco_labels.txt')#required=True, 
    parser.add_argument(
            '--top_k', help="keep top k candidates.", default=5)
    parser.add_argument(
            '--threshold', help="threshold to filter results.", default=0.3)
    parser.add_argument(
            '--width', help="Resolution width.", default=800)
    parser.add_argument(
            '--height', help="Resolution height.", default=800)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine.
    engine = edgetpu.detection.engine.DetectionEngine(args.model)
    labels = ReadLabelFile(args.label) if args.label else None

    # Generate random colors.
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    colors = visual.random_colors(last_key)

    elapsed_list = []
    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(rawCapture,
                                                 format='rgb',
                                                 use_video_port=True):
                rawCapture.truncate(0)

                # input_buf = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                input_buf = PIL.Image.fromarray(image)

                # Run inference.
                start_ms = time.time()
                ans = engine.DetectWithImage(input_buf, threshold=args.threshold,
                       keep_aspect_ratio=False, relative_coord=False, top_k=args.top_k)
                # ans = engine.DetectWithInputTensor(input_buf, threshold=0.05,
                #         keep_aspect_ratio=False, relative_coord=False, top_k=10)
                elapsed_ms = time.time() - start_ms

                # Display result.
                if ans:
                    for obj in ans:
                        label_name = 'Unknown'
                        if labels:
                            label_name = labels[obj.label_id]
                        caption = '{0}({1:.2f})'.format(label_name, obj.score)

                        # Draw a rectangle and caption.
                        box = obj.bounding_box.flatten().tolist()
                        visual.draw_rectangle(im, box, colors[obj.label_id])
                        visual.draw_caption(im, box, caption)

                # Calc fps.
                fps = 1 / elapsed_ms
                elapsed_list.append(elapsed_ms)
                avg_text = ""
                if len(elapsed_list) > 100:
                    elapsed_list.pop(0)
                    avg_elapsed_ms = np.mean(elapsed_list)
                    avg_fps = 1 / avg_elapsed_ms
                    avg_text = ' AGV: {0:.2f}ms, {1:.2f}fps'.format(
                        (avg_elapsed_ms * 1000.0), avg_fps)

                # Display fps
                fps_text = '{0:.2f}ms, {1:.2f}fps'.format(
                        (elapsed_ms * 1000.0), fps)
                visual.draw_caption(im, (10, 30), fps_text + avg_text)

                # display
                cv2.imshow(WINDOW_NAME, im)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()