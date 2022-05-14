# tflite-pi-test
Run object-detection NN models on Raspberry Pi 4B, and quantize models with TensorFlow Lite.

## Install TensorFlow
Install TensorFlow using Tensorflow-bin.  
Tensorflow-bin provides Prebuilt binary with Tensorflow Lite enabled for RaspberryPi.

Follow the repository's usage, install `tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1200`.

## Download models
Use the neural network models in TensorFlow's Object Detection API.  

This project tests `CenterNet MobileNetV2 FPN 512x512` and `EfficientDet D0 512x512` in [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

- Download the models
```sh
$ cd src

$ wget http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
$ tar xvzf centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
$ rm centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz

$ wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
$ tar xvzf efficientdet_d0_coco17_tpu-32.tar.gz
$ rm efficientdet_d0_coco17_tpu-32.tar.gz
```
- Check model structures and input data types, shapes etc.
```sh
$ saved_model_cli show --dir ./centernet_mobilenetv2_fpn_od/saved_model/ --all
$ saved_model_cli show --dir ./efficientdet_d0_coco17_tpu-32/saved_model/ --all
```

## Run
Run with the scripts. This script is based on [this page](https://qiita.com/karaage0703/items/8c3197d11f61812546a9).

- Inference
  - Output object-detection result's image (`result.png`) in the current directory.
```sh
$ python inference.py --saved_model=centernet_mobilenetv2_fpn_od --num_run=50
median:102.1 / mean :121.3 [msec]
max score: 0.6
```
- Quantize models
  - `model.tflite` is [Post-training float16 quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant) model
  - > Note how the resulting file is approximately 1/2 the size.
```sh
$ python convert_to_tflite.py --saved_model=centernet_mobilenetv2_fpn_od --output=model.tflite
```
- Inference with tflite file
```sh
$ python inference_lite.py --saved_model=model.tflite --num_run=50
```
