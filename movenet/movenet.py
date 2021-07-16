import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import colors
from utils import  helper_function

model_name = "movenet_thunder"

if "tflite" in model_name:
  if "movenet_lightning" in model_name:
    input_size = 192
    model_path = 'models/movenet_singlepose_lightning_3.tflite'
  elif "movenet_thunder" in model_name:
    input_size = 256
    model_path = 'models/movenet_singlepose_thunder_3.tflite'
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in model_name:
      module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
      input_size = 192
  elif "movenet_thunder" in model_name:
      module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
      input_size = 256
  else:
      raise ValueError("Unsupported model name: %s" % model_name)

  def movenet(input_image):
      """Runs detection on an input image.

      Args:
        input_image: A [1, height, width, 3] tensor represents the input image
          pixels. Note that the height/width should already be resized and match the
          expected input resolution of the model before passing into this function.

      Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
      """
      model = module.signatures['serving_default']

      # SavedModel format expects tensor type of int32.
      input_image = tf.cast(input_image, dtype=tf.int32)
      # Run model inference.
      outputs = model(input_image)
      # Output is a [1, 1, 17, 3] tensor.
      keypoint_with_scores = outputs['output_0'].numpy()
      return keypoint_with_scores

def preprocess(image,size):

    input_w = size
    input_h = size
    h,w = image.shape[:2]
    if (h > w):
            left = right = int((h - w) / 2)
            top = bottom = 0
    else:
        top = bottom = int((w - h) / 2)
        left = right = 0

    full_orig_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
    
    full_orig_image = cv2.resize(full_orig_image, (input_w, input_h), interpolation=cv2.INTER_LANCZOS4)
    
    RGB_img = cv2.cvtColor(full_orig_image,cv2.COLOR_BGR2RGB)

    return RGB_img, left, right, top, bottom

def reverse_transform(mask,h,w,pad):

    l,r,t,b =  pad
    a = cv2.resize((mask).astype('float'),(w+r +l,h+t +b))
    h,w  =  a.shape[:2]
    a = a[t:h-b,l:w -r]

    return a.astype(np.uint8)

def run_inference_deeplab(img, deep_lab_model):

    h,w = img.shape[:2]
    processed_img, left, right, top, bottom = preprocess(img)

    _, mask  =  deep_lab_model.run(processed_img)

    alpha  = reverse_transform(mask,h,w,(left,right,top,bottom))
    alpha = (alpha*255).astype(np.uint8)

    return alpha

def resize_pad(image,height,width):
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image,height,width), dtype=tf.int32)

    display_image = np.squeeze(display_image.numpy(), axis=0)
    return display_image

# main function
if __name__ == "__main__":
  image  = cv2.imread("static/pexels_Woman.jpg")

  image_height, image_width, _ = image.shape
  crop_region = helper_function.init_crop_region(image_height, image_width)

  keypoints_with_scores = helper_function.run_inference_frame(
        movenet, image, crop_region,
        crop_size=[input_size, input_size])
  (keypoint_locs, keypoint_edges,edge_colors) = helper_function._keypoints_and_edges_for_display(keypoints_with_scores, image_height, image_width)
  test_image = image.copy()
  test_image =helper_function.draw_prediction_on_image_cv(test_image,keypoint_locs, keypoint_edges,edge_colors)

  cv2.imwrite("static/Movenet_output.jpg",test_image)
  cv2.imshow('image', test_image)
  if cv2.waitKey(0) == 27:
      cv2.destroyAllWindows()