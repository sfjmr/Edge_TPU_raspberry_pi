"""A demo to classify Raspberry Pi camera stream."""
import argparse
import io
import time

import numpy as np
import picamera

import edgetpu.classification.engine
import edgetpu.detection.engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    #engine = edgetpu.classification.engine.ClassificationEngine(args.model)
    engine = edget pu.detection.engine.DetectionEngine(args.model)
    print("engine.required_input_array_size()")
    print(engine.required_input_array_size())
    with picamera.PiCamera() as camera:
        #camera.resolution = (640, 480)
        camera.resolution = (300, 900)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        print("width, height, channels")
        print(width, height, channels)
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream,
                                                 format='rgb',
                                                 use_video_port=True,
                                                 resize=(width, height)):
                stream.truncate()
                stream.seek(0)
                input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                print("input.shape")
                print(input.shape)
                start_ms = time.time()
                #results = engine.ClassifyWithInputTensor(input, top_k=1)
                results = engine.DetectWithInputTensor(input, threshold=0.1, top_k=3)
                elapsed_ms = time.time() - start_ms
                if results:
                    #結果を表示
                    #camera.annotate_text = "%s %.2f\n%.2fms" % (
                    #    labels[results[0][0]], results[0][1], elapsed_ms*1000.0)
                    pass


        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()
