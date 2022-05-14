import os
import tensorflow as tf
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

def make_input_tensor_from_image(image_filename, saved_model_name):
  # read image
  img = cv2.imread(image_filename)

  # convert bgr to rgb
  img_bgr = cv2.resize(img, (320, 320)) # resize ot 320x320
  image_np = img_bgr[:,:,::-1]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  input_data = tf.convert_to_tensor(image_np_expanded)
  if saved_model_name == 'centernet_mobilenetv2_fpn_od':
    input_data = tf.dtypes.cast(input_data, tf.float32)
  
  return img, input_data

def parse_centernet_mobilenetv2_fpn_od_output(output_data):
  num_detections =      int(output_data['output_0'][0])
  scores         = np.array(output_data['output_1'][0])
  classes        = np.array(output_data['output_2'][0], dtype=np.int32)
  boxes          = np.array(output_data['output_3'][0])
  return num_detections, scores, classes, boxes

def parse_efficientdet_d0_coco17_tpu(output_data):
  num_detections =      int(output_data['num_detections'][0])
  scores         = np.array(output_data['detection_scores'][0])
  classes        = np.array(output_data['detection_classes'][0], dtype=np.int32)
  boxes          = np.array(output_data['detection_boxes'][0])
  return num_detections, scores, classes, boxes

def parse_ssd_mobilenet_v2(output_data):
  num_detections =      int(output_data['num_detections'][0])
  scores         = np.array(output_data['detection_scores'][0])
  classes        = np.array(output_data['detection_classes'][0], dtype=np.int32) - 1
  boxes          = np.array(output_data['detection_boxes'][0])
  return num_detections, scores, classes, boxes

def parse_output_data(saved_model_name, output_data):
  if saved_model_name == 'efficientdet_d0_coco17_tpu-32':
    return parse_efficientdet_d0_coco17_tpu(output_data)
  elif saved_model_name == 'centernet_mobilenetv2_fpn_od':
    return parse_centernet_mobilenetv2_fpn_od_output(output_data)
  elif 'ssd_mobilenet_v2' in saved_model_name:
    return parse_ssd_mobilenet_v2(output_data)
  else:
    assert False, 'not implemented'

def draw_result(img, labels_list, num_detections, scores, classes, boxes, score_th, savename='result.png'):
  # draw results
  for i in range(num_detections):
    score = scores[i]
    class_id = classes[i]
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

def main():
  parser = argparse.ArgumentParser(description='Inference sample of TensorFlow 2 Detection Model Zoo')
  parser.add_argument('--saved_model', default='centernet_mobilenetv2_fpn_od', help='saved model directory name (without /saved_model)')
  parser.add_argument('--input_image', default='sample.jpg', help='input image file name')
  parser.add_argument('--labels_text', default='coco-labels-paper.txt', help='COCO labels text file name')
  parser.add_argument('--num_run', default=1, type=int, help='the number of run')
  parser.add_argument('--score_th', default=0.3, type=float, help='threshold of score')
  args = parser.parse_args()
  saved_model_name = args.saved_model

  # read labels text
  labels_list = read_labels_list(args.labels_text)

  # load model
  saved_model_path = os.path.join(saved_model_name, 'saved_model')
  loaded = tf.saved_model.load(saved_model_path)
  inference_func = loaded.signatures["serving_default"]

  # set up input tensor
  img, input_data = make_input_tensor_from_image(args.input_image, saved_model_name)

  # inference
  elapsed = []
  for _ in range(args.num_run):
    begin_t = time.perf_counter()
    output_data = inference_func(input_data)
    end_t = time.perf_counter()
    elapsed.append(end_t - begin_t)
  elapsed_ms = np.array(elapsed) * 1000
  print(f'median:{np.median(elapsed_ms):5.1f} / mean :{np.mean(elapsed_ms):5.1f} [msec]')

  # get num_detections, scores, classes, boxes
  num_detections, scores, classes, boxes = parse_output_data(saved_model_name, output_data)

  # draw bounding-box
  print(f'max score: {np.max(scores):.1f}')
  thresehold = args.score_th
  draw_result(img, labels_list, num_detections, scores, classes, boxes, thresehold)

if __name__ == '__main__':
  main()

# References
# - https://qiita.com/karaage0703/items/8c3197d11f61812546a9
