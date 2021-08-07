#!/usr/bin/env python3

import rospy
import numpy as np
import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import deeplab.deeplab_ros as dlros

class DeeplabNode():

    def __init__(self):

        #Initialize variables
        self.bridge = CvBridge()
        
        # create model from model config 
        self.deeplabModel = dlros.DeepLabModel(dlros.DeepLabModelConfig(rospy.get_param("deeplab_model")))
        
        #Subscriber (buff_size is set to 2**24 to avoid delays in the callbacks)
        self.sub = rospy.Subscriber("image", Image, self.imageCallback, queue_size=1, buff_size=2**24)

        #Publisher
        self.pub = rospy.Publisher("output_image", Image, queue_size=1)
        self.pub_seg = rospy.Publisher("output_segmap", Image, queue_size=1)

    def imageCallback(self, data): #Function that runs when an image arrives
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data) # Transforms the format of image into OpenCV 2
            segmentation_map = self.deeplabModel.segmentImage(cv_image, True)
            color_map = self.deeplabModel.getColormapFromSegmentationMap(segmentation_map).astype('uint8')

            labels, unique_idxs = np.unique(segmentation_map, return_index = True)

            print('\033[2J')
            for (label, unique_idx) in zip(labels, unique_idxs):
                print("(class: {}".format(self.deeplabModel.modelConfig.detectionClasses[label]), self.deeplabModel.getColormapFromSegmentationMap(np.array([[label]])))

            self.pub_seg.publish(self.bridge.cv2_to_imgmsg(color_map, encoding=data.encoding))
            self.pub.publish(self.bridge.cv2_to_imgmsg(cv2.addWeighted(cv_image, 0.33, color_map, 0.66, 0).astype('uint8'), encoding=data.encoding))
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
            return


def main():

    rospy.init_node('deeplab_segmentation')
    hd = DeeplabNode()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("error")
        pass
    print("Exiting process...")