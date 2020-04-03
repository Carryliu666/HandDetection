#-*- coding:utf8 -*-
import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO

from PIL import Image

import matplotlib.pyplot as plt
# Import the object detection module.


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile



# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'training/object_detection.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# model_name='ssdmo_v1_1class_inference_graph'
def load_model(model_name):

  # Maybe a problem will ocur because belowed "saved_model"
  # Dont know "saved_model" point to the dir or saved_model.pb
  model_dir = pathlib.Path(model_name)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


# if __name__ == '__main__':

model_name='ssdmo_v1_1class_inference_graph'
detection_model =load_model(model_name)
print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        output_dict['detection_masks'], output_dict['detection_boxes'],
        image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                         tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  plt.imshow(Image.fromarray(image_np))
  plt.show()
def video_show_inference(model, image_np):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  plt.imshow(Image.fromarray(image_np))
  plt.show()


cap = cv2.VideoCapture(0)
width = 1280
height = 720
w = 360
#设置摄像头捕获的分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#设置要保存的视频编码,分辨率和帧率
video = cv2.VideoWriter("HandVideo.avi",cv2.VideoWriter_fourcc('M','P','4','2'),20,(1280,720))
while(cap.isOpened()):
    #成功则ret为True, frame是数组
    ret,frame=cap.read()
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
    video_show_inference(detection_model,image_np)

    cv2.imshow("Video",frame)

    if cv2.waitKey(10)==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
# for image_path in TEST_IMAGE_PATHS:
#  show_inference(detection_model, image_path)

