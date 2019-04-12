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
    engine = edgetpu.detection.engine.DetectionEngine(args.model)
    print("engine.required_input_array_size()")
    print(engine.required_input_array_size()) #270000
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        #300 300 291840
        #270 270 291840
        #camera.resolution = (270, 270)
        camera.framerate = 30 #20でも変化なし
        _, width, height, channels = engine.get_input_tensor_shape()
        print("width, height, channels")
        print(width, height, channels)
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream,
                                                 format='rgb',
                                                 use_video_port=True,
                                                 resize=(256, 256)):#うまくresizeされてないのが問題?
                #300 300 -> 291840
                #250 250 -> 196608
                #200 200 -> 139776
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
