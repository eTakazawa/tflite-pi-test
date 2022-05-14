import tensorflow as tf
from tensorflow.lite.python.interpreter import load_delegate
import cv2
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

def read_labels_list(labels_text_path):
  labels_list = []
  with open(labels_text_path) as labels:
    for line in labels:
      labels_list.append(line.strip())
  return labels_list

def make_input_tensor_from_image(image_filename, input_details):
  # read property
  floating_point = not input_details[0]['dtype'] == np.uint8
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  # read image
  img = cv2.imread(image_filename)

  # convert bgr to rgb
  img_bgr = cv2.resize(img, (width, height))
  image_np = img_bgr[:,:,::-1]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  input_data = tf.convert_to_tensor(image_np_expanded)
  if floating_point:
    input_data = tf.dtypes.cast(input_data, tf.float32)
  
  return img, input_data

def sort_by_name(interpreter, output_details):
  output_data = {}
  for detail in output_details:
    n = detail['name'].split(':')[1]
    output_data[f'output_{n}'] = interpreter.get_tensor(detail['index'])
  return output_data

def parse_centernet_mobilenetv2_fpn_od_output_lite(interpreter, output_details):
  output_data = sort_by_name(interpreter, output_details)
  num_detections =      int(output_data['output_0'][0])
  scores         = np.array(output_data['output_1'][0])
  classes        = np.array(output_data['output_2'][0], dtype=np.int32)
  boxes          = np.array(output_data['output_3'][0])
  return num_detections, scores, classes, boxes

def parse_ssd_mobilenet_v2(interpreter, output_details):
  output_data = sort_by_name(interpreter, output_details)
  num_detections =      int(output_data['output_5'][0])
  scores         = np.array(output_data['output_4'][0])
  classes        = np.array(output_data['output_2'][0], dtype=np.int32) - 1
  boxes          = np.array(output_data['output_1'][0])
  return num_detections, scores, classes, boxes

def parse_efficientdet_d0_coco17_tpu_lite(interpreter, output_details):
  output_data = sort_by_name(interpreter, output_details)
  num_detections =      int(output_data['output_5'][0])
  scores         = np.array(output_data['output_4'][0])
  classes        = np.array(output_data['output_2'][0], dtype=np.int32)
  boxes          = np.array(output_data['output_1'][0])
  return num_detections, scores, classes, boxes

def draw_result(img, labels_list, num_detections, scores, classes, boxes, score_th, savename='result.png'):
  # draw results
  for i in range(num_detections):
    score = scores[i]
    class_id = classes[i]
    # print(class_id, score)
    if score > score_th:
      h, w, _ = img.shape
      box = (boxes[i] * np.array([h, w, h, w])).astype(np.int32)
      # draw bounding box
      cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)
      cv2.putText(img, labels_list[class_id], (box[1], box[0]-8), cv2.FONT_HERSHEY_COMPLEX_SMALL | cv2.FONT_ITALIC, 2, (0, 0, 255), 1, cv2.LINE_AA)

  # show
  img_rgb = img[:,:,::-1]
  plt.axis("off")
  plt.imshow(img_rgb)
  plt.savefig(savename, transparent = True, bbox_inches = 'tight', pad_inches = 0)
  print(f'save the image to {savename}')

def main():
  parser = argparse.ArgumentParser(description='Inference tflite')
  parser.add_argument('--saved_model', default='model.tflite', help='tflite')
  parser.add_argument('--input_image', default='sample.jpg', help='input image file name')
  parser.add_argument('--labels_text', default='coco-labels-paper.txt', help='COCO labels text file name')
  parser.add_argument('--num_threads', default=1, type=int, help='[tf.lite.Interpreter option] num_threads')
  parser.add_argument('--num_run', default=1, type=int, help='the number of run')
  parser.add_argument('--delegate', default='', type=str, help='armnn delegate object file')
  parser.add_argument('--score_th', default=0.1, type=float, help='threshold of score')
  args = parser.parse_args()

  # read labels text
  labels_list = read_labels_list(args.labels_text)

  # load tflite model
  delegate = [load_delegate(args.delegate)] if args.delegate != '' else None
  interpreter = tf.lite.Interpreter(
    model_path=args.saved_model,
    num_threads=args.num_threads, #experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
    experimental_delegates=delegate
    )
  interpreter.allocate_tensors()

  # get input/output details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # setup input tensor
  img, input_data = make_input_tensor_from_image(args.input_image, input_details)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  np.save('input_data', input_data)

  # inference
  elapsed = []
  for _ in range(args.num_run):
    begin_t = time.perf_counter()
    interpreter.invoke()
    end_t = time.perf_counter()
    elapsed.append(end_t - begin_t)
  elapsed_ms = np.array(elapsed) * 1000
  print(f'median:{np.median(elapsed_ms):5.1f} / mean :{np.mean(elapsed_ms):5.1f} [msec]')

  if 'centernet' in args.saved_model:
    num_detections, scores, classes, boxes = parse_centernet_mobilenetv2_fpn_od_output_lite(interpreter, output_details)
  elif 'ssd' in args.saved_model:
    num_detections, scores, classes, boxes = parse_ssd_mobilenet_v2(interpreter, output_details)
  else:
    num_detections, scores, classes, boxes = parse_efficientdet_d0_coco17_tpu_lite(interpreter, output_details)

  # draw results
  print(f'max score: {np.max(scores):.1f}')
  thresehold = args.score_th # 0.1 # 0.3
  draw_result(img, labels_list, num_detections, scores, classes, boxes, thresehold, 'result_lite.png')

if __name__ == '__main__':
  main()

# Reference
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
