#/usr/bin/env python3

# Copyright 2021 ROS-Force
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED 
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==============================================================================

import numpy as np
import cv2
import tensorflow as tf # needs TF 2.x version 
import deeplab.get_dataset_colormap as dlutils
physical_devices = tf.config.list_physical_devices('GPU')

try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("No GPU device was found")
class DeepLabModel(object):
  """Class to load DeepLab model and run inference on supplied images."""

  def __init__(self, modelConfig):
    """Creates and loads pretrained DeepLab model."""

    # grab model config
    self.modelConfig = modelConfig
 
    with open(self.modelConfig.graphPath, 'rb') as file_handle:
      # load frozen inference graph from file and wrap it into a tf.WrapperFunction
      self.modelFunction = DeepLabModel.__wrap_frozen_graph(tf.compat.v1.GraphDef.FromString(file_handle.read()), self.modelConfig.inputTensorName, self.modelConfig.outputTensorName)

  def segmentImage(self, image, isImageRGB=True):
    """Runs inference on a single image.

    Args:
      image: A OpenCV RGB image.

    Returns:
      result: Segmentation map of the input image (for OpenCV).
          
    TODO: 
    - Adjust model input & output according to the supplied param (modelConfig.outputTensorName & modelConfig.inputTensorName)
    - Check how the image preprocessing affects other rosbags
    """ 

    # get target size 
    image_shape = np.array(image.shape[:-1])
    target_size = (min(self.modelConfig.inputSize.astype(np.float64) / image_shape) * image_shape).astype(np.uint32)

    # resize and apply antialias filter
    input_image = tf.image.resize(tf.convert_to_tensor(image, dtype=tf.dtypes.uint8), target_size, method=tf.image.ResizeMethod.AREA)

    # check if image needs to be swaped to RGB / BGR
    if (isImageRGB ^ (not self.modelConfig.inputBGR)):
      input_image = tf.reverse(input_image, axis=[-1])

    # add extra dimension and convert array to tf.Tensor (or more accurately, to an ImageTensor)
    segmentation_map = tf.squeeze(self.modelFunction(tf.expand_dims(tf.cast(input_image, dtype=tf.uint8), axis=0)))

    # resize to original size and convert to opencv image (NOTE: shape is used in REVERSED on opencv)
    return cv2.resize(segmentation_map.numpy().astype(np.float32), image.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    
    
  def segmentImage_tf(self, image, isImageRGB=True):
    """Runs inference on a single image.

    Args:
      image: A OpenCV RGB image.

    Returns:
      result: Segmentation map of the input image (for TF).
          
    TODO: Adjust model input & output according to the supplied param (modelConfig.outputTensorName & modelConfig.inputTensorName)
    """ 
    

    # get target size
    image_shape = np.array(image.shape[:-1])
    target_size = (min(self.modelConfig.inputSize.astype(np.float64) / image_shape) * image_shape).astype(np.uint32)

    # resize and apply antialias filter
    input_image = tf.image.resize(tf.convert_to_tensor(image, dtype=tf.dtypes.uint8), target_size, method=tf.image.ResizeMethod.AREA)

    # check if image needs to be swaped to RGB / BGR
    if (isImageRGB ^ (not self.modelConfig.inputBGR)):
      input_image = tf.reverse(input_image, axis=[-1])

    # add extra dimension and convert array to tf.Tensor (or more accurately, to an ImageTensor)
    segmentation_map = tf.squeeze(self.modelFunction(tf.expand_dims(tf.cast(input_image, dtype=tf.uint8), axis=0)))

    # return reshaped segmentation map to fit original image
    return tf.cast(tf.image.resize(segmentation_map, image.shape[1::-1], method=tf.ResizeMethod.NEAREST), dtype=tf.uint8)
  
  def getColormapFromSegmentationMap(self, segmentation_map):
    """Adds color defined by the dataset colormap to the label.

    Args:
      segmentation_map: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the dataset color map.

    """

    return dlutils.label_to_color_image(segmentation_map, self.modelConfig.datasetName)
  
  @staticmethod
  def __wrap_frozen_graph(graph_def, inputs, outputs):
    """ 
    Wrapper for frozen inference graph (works as a compatibility layer between TF 1.x and TF 2.x). 
    
    Args:
      graph_def: Represents the graph of operations.
      inputs: Text description of the input tensors of the graph.
      outputs: Text description of the input tensors of the graph.

    Returns:
      result: tf.WrappedFunction of the imported graph.
      
    """

    wrapped_import = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(graph_def, name=""), [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

class DeepLabModelConfig(object):
  '''
  Helper class to hold the DeepLab model's parameters. 
  '''

  def __init__(self, graphPath, datasetName, inputTensorName, inputBGR, outputTensorName, inputSize):
    self.graphPath = graphPath
    self.datasetName = datasetName
    self.inputBGR = inputBGR
    self.inputTensorName = inputTensorName
    self.outputTensorName = outputTensorName
    self.inputSize = np.array(inputSize)
  
  def __init__(self, paramDict):
    self.graphPath = paramDict.get('inference_graph', {}).get('path')
    self.inputTensorName = paramDict.get('inference_graph', {}).get('input_tensor')
    self.outputTensorName = paramDict.get('inference_graph', {}).get('output_tensor')
    self.inputBGR = paramDict.get('input_bgr')
    self.inputSize = np.array(paramDict.get('input_size', []))
    self.datasetName = paramDict.get('dataset', {}).get('name')
    self.detectionClasses = paramDict.get('dataset', {}).get('detection_classes')