import os
import argparse
import tensorflow as tf
import numpy as np

def representative_dataset():
  # data = np.load('input_data.npy')
  for _ in range(100):
    data = np.random.rand(1, 320, 320, 3) * 128
    yield [data.astype(np.uint8)]

def main():
  parser = argparse.ArgumentParser(description='Inference sample of TensorFlow 2 Detection Model Zoo')
  parser.add_argument('--saved_model', default='centernet_mobilenetv2_fpn_od', help='saved model directory (without /saved_model)')
  parser.add_argument('--output', default='model.tflite', help='output tflite filename')
  args = parser.parse_args()
  
  saved_model_name = args.saved_model
  saved_model_path = os.path.join(saved_model_name, 'saved_model')

  # Convert to tflite
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS    # enable TensorFlow ops.
  ]
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  # converter.representative_dataset = representative_dataset
  tflite_model = converter.convert()

  tflite_model_name = args.output
  with open(tflite_model_name, 'wb') as o_:
    o_.write(tflite_model)
    print(f'convert to {tflite_model_name}')

if __name__ == '__main__':
  main()

