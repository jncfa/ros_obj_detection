#!/usr/bin/env python3

from numpy.core.fromnumeric import mean
import rospy
import numpy as np
import cv2
from colorutils import Color
import sys

from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from object_detection.msg import BoundingBox, BoundingBoxes, TrackerBox, TrackerBoxes, CenterID, Object_info
from geometry_msgs.msg import Vector3
import message_filters

from sort.sort import Sort
class SortTrackingNode():

    def __init__(self):

        #Initialize variables
        self.IoU_THRESHOLD = rospy.get_param("~sort/threshold", 0.1) #Minimum IOU for match
        self.MIN_HITS = rospy.get_param("~sort/min_hits", 3) #Minimum number of associated detections before track is initialised
        self.MAX_AGE = rospy.get_param("~sort/max_age", 20) #Maximum number of frames to keep alive a track without associated detections.
        self.BLUR_HUMANS = rospy.get_param("~blur_humans", False)
        self.DRAW_SPEED = rospy.get_param("~draw_speed", False)


        self.previous_time = rospy.Time.now()
        self.previous_centers = {}
        self.cameraInfo = None

        self.bridge = CvBridge()
        
        self.mo_tracker = Sort(max_age=self.MAX_AGE, min_hits=self.MIN_HITS, iou_threshold=self.IoU_THRESHOLD) #Create the multi-object tracker

        # check which yolo model is loaded
        
        yoloModelPrefix = [param for param in rospy.get_param_names() if "model_prefix" in param][0]
        if (yoloModelPrefix):
            self.classes = rospy.get_param(rospy.get_param(yoloModelPrefix) + "/dataset/detection_classes")
        else:
            rospy.logerr("Could not load detection classes from YOLO model, exiting.")
            sys.exit(1)

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        #Publishers
        self.pub = rospy.Publisher("output_image", Image, queue_size=10)
        self.pub_obj = rospy.Publisher("object", Object_info, queue_size=10)

        #Subscriber
        self.sub = message_filters.Subscriber("image", Image, queue_size=100, buff_size=2**27)
        self.sub_depth = message_filters.Subscriber("depth_image", Image, queue_size=100, buff_size=2**27)
        self.sub_bbox_yolo  = message_filters.Subscriber("bounding_boxes", BoundingBoxes, queue_size=100, buff_size=2**27)
        
        ts = message_filters.ApproximateTimeSynchronizer([self.sub, self.sub_depth, self.sub_bbox_yolo], 60, 0.1)
        ts.registerCallback(self.imageCallback)
        
        # grab camera info
        self.imageDepthInfoCallback(rospy.wait_for_message("camera_info", CameraInfo))

#Callbacks
    def imageCallback(self, img_data, depth_data, list_bbox): #Function that runs when a RGB image arrives
        self.data_encoding = img_data.encoding #Stores the encoding of original image
        cv_image = self.bridge.imgmsg_to_cv2(img_data, img_data.encoding)  # Transforms the format of image into OpenCV 2
        
        data_encoding_depth = depth_data.encoding
        cv_image_depth = self.bridge.imgmsg_to_cv2(depth_data, depth_data.encoding) # Transforms the format of image into OpenCV 2

        if(cv_image.shape != cv_image_depth.shape):
            cv_image = cv2.resize(cv_image, (cv_image_depth.shape[1],cv_image_depth.shape[0]))
        center_list = {}

        if cv_image is not None and cv_image_depth is not None: # if RGB and Depth images are available runs the code

            img = np.copy(cv_image)
            img_depth = np.copy(cv_image_depth)
            trackers = self.mo_tracker.update(list_bbox) #Update the Tracker Boxes positions
            for t in trackers.tracker_boxes: #Go through every detected object
                
                obj = Object_info()
                center_pose = self.computeRealCenter(t, img_depth) #compute the Real pixels values
                
                speed = self.computeSpeed(center_pose, t.id, trackers.header.stamp) #compute the velocity vector of the diferent objects
                img, color, sat, ilu, shape = self.computeFeatures(t, img, img_depth) #compute diferent features, like color and shape
                center_list[t.id] = center_pose #add this center to the dictionary of centers
                if speed is not None:
                    #Construct the object with its atributes
                    obj.id = t.id
                    obj.Class = t.Class
                    obj.real_pose = center_pose
                    obj.velocity = speed
                    obj.bbox = [int(t.xmin), int(t.ymin), int(t.xmax), int(t.ymax)]
                    obj.color = color
                    obj.saturation = sat
                    obj.ilumination = ilu 
                    obj.shape = shape
                    img = self.draw_labels(obj, img) #draws the labels in the original image
                
                self.pub_obj.publish(obj) # publish the object

                self.previous_centers = center_list #set the current center dictionary as previous dictionary
                self.previous_time = trackers.header.stamp

            image_message = self.bridge.cv2_to_imgmsg(img, encoding = self.data_encoding)
            self.pub.publish(image_message) #publish the labelled image

    def imageDepthInfoCallback(self, cameraInfo): #Code copied from Intel script "show_center_depth.py". Gather camera intrisics parameters that will be use to compute the real coordinates of pixels
        self.cameraInfo = cameraInfo

    def computeRealCenter(self, tracker, cv_image_depth):
        
        center = Vector3()

        pix = [int(tracker.xmin + (tracker.xmax-tracker.xmin)//2), int(tracker.ymin + (tracker.ymax-tracker.ymin)//2)] #Coordinates of the central point (in pixeis)
        depth = cv_image_depth[pix[1], pix[0]] #Depth of the central pixel
        #result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth) # Real coordenates, in mm, of the central pixel
        result = SortTrackingNode.deproject_pixel_to_point(self.cameraInfo, [pix[0], pix[1]], depth) # Real coordenates, in mm, of the central pixel
        
        #Create a vector with the coordinates, in meters
        center.x = result[0]*10**(-3)
        center.y = result[1]*10**(-3)
        center.z = result[2]*10**(-3)

        return center


    def computeSpeed(self, center, id, new_time):

        speed = Vector3()

        if len(self.previous_centers) == 0:
            
            return None
        
        else:
            
            deltat = (new_time - self.previous_time) #Compute the diference in times between the previous time and this frame, 
                                                    # The time of the current tracker was created when the image arrived (in 
            deltat = deltat.to_sec()                 # the code "yolo_detection.py" inside the imageCallback() function).

            
            previous_center = self.previous_centers.get(id) #get the center of previous object with that id, if the id isn't fouund return a None that is treated in the next IF

            if previous_center is not None: #If exists a center in the last frame computes the speed
                    
                speed.x = (center.x - previous_center.x)/deltat
                speed.y = (center.y - previous_center.y)/deltat
                speed.z = (center.z - previous_center.z)/deltat

            return speed #in meters/sec

    def computeFeatures(self, t, img, img_depth):

        thr_img, depth_thresh, roi = self.computeRoi(img, img_depth, t) #select the roi of the object

        sat = mean(roi[:,:,1])/255.0 # sat mean of image (E PARA MUDAR ISTO)
        ilumination = mean(roi[:,:,2])/255.0

        if(t.Class == "person" and self.BLUR_HUMANS):

            img[int(t.ymin):int(t.ymax), int(t.xmin):int(t.xmax)] = cv2.blur(img[int(t.ymin):int(t.ymax), int(t.xmin):int(t.xmax)] ,(25,25))


        if(t.Class == "traffic light"):
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV) # change image from rgb to hsv

            mask1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
            
            mask2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
            maskg = cv2.inRange(hsv, np.array([40,50,50]), np.array([90,255,255]))
            masky = cv2.inRange(hsv, np.array([15,150,150]), np.array([35,255,255]))
            maskr = cv2.add(mask1, mask2)
            count_red = np.count_nonzero(maskr)
            count_green = np.count_nonzero(maskg)
            count_yellow = np.count_nonzero(masky)

            if(count_red >= count_yellow and count_red>= count_green):
                color = "Red"
            elif(count_green >= count_yellow and count_green > count_red):
                color = "Green"
            else:
                color="Yellow"
            
        else:
            [counts, values] = np.histogram(thr_img[:,:,0], bins=36, range=(1,180)) #create an histogram to see the most present color
            m = max(counts)
            dic = dict(zip(counts, values))
            hue = int(dic[m] * 2) # hue valor of the most present color
            image_message = self.bridge.cv2_to_imgmsg(img, encoding = self.data_encoding)
            self.pub.publish(image_message) #publish the labelled image

            c = Color(hsv=(hue, sat, ilumination))
            color = c.web

        shape = self.computeShape(depth_thresh)

        return img, color, int(sat*100), int(ilumination*100), shape 


    def computeShape(self, thresh):

        contours, hierarchy = cv2.findContours(self.map_uint16_to_uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = contours[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1*peri, True)

        if len(approx) == 3:
            shape = "triangle"

        elif len(approx) == 4:

            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        elif len(approx) == 5:
            shape = "pentagon"

        else:
            shape = "circle"

        return shape


    def computeRoi(self, img, depth, t):

        roi = img[int(t.ymin):int(t.ymax), int(t.xmin):int(t.xmax)] #select the region of interess of rgb and depth images
        roi_depth = depth[int(t.ymin):int(t.ymax), int(t.xmin):int(t.xmax)]
        
        m = mean(roi_depth)
        std = np.std(roi_depth)
        
        ret, thresh = cv2.threshold(roi_depth, m, np.amax(roi_depth),cv2.THRESH_BINARY_INV) #threshold to try to minimize the backgound influence
        mask = self.map_uint16_to_uint8(thresh, lower_bound=0, upper_bound=255)
        thr_img = cv2.bitwise_and(roi, roi,mask = mask) #apply the mask created to the rgb image
        thr_img = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        return thr_img, thresh, roi


    def map_uint16_to_uint8(self, img, lower_bound=None, upper_bound=None):
        if lower_bound is not None and not(0 <= lower_bound < 2**16):
            raise ValueError(
                '"lower_bound" must be in the range [0, 65535]')
        if upper_bound is not None and not(0 <= upper_bound < 2**16):
            raise ValueError(
                '"upper_bound" must be in the range [0, 65535]')
        if lower_bound is None:
            lower_bound = np.min(img)
        if upper_bound is None:
            upper_bound = np.max(img)
        if lower_bound > upper_bound:
            raise ValueError(
                '"lower_bound" must be smaller than "upper_bound"')
        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)

    def draw_labels(self, obj, img): 
        font = cv2.FONT_HERSHEY_PLAIN


        label = "%s #%i" % (obj.Class, obj.id)
        color = self.colors[int(obj.id) % len(self.colors)]
        cv2.rectangle(img, (int(obj.bbox[0]), int(obj.bbox[1]), int(obj.bbox[2]) - int(obj.bbox[0]), int(obj.bbox[3]) - int(obj.bbox[1])), color, 2) #draws a rectangle in the original image
        cv2.putText(img, label, (int(obj.bbox[0]), int(obj.bbox[1] - 10)), font, 1, color, 1) #writes the Class and Object ID in the original image
        
        if(self.DRAW_SPEED):
            speed_label = "Vx = %.1f; Vy = %.1f; Vz = %.1f; " % (obj.velocity.x, obj.velocity.y, obj.velocity.z)
            cv2.putText(img, speed_label, (int(obj.bbox[0]), int(obj.bbox[3] + 10)), font, 1, color, 1) #writes the Class and Object ID in the original image

        return img

    @staticmethod
    def deproject_pixel_to_point(cameraInfo, pixel, depth):
        # parse intrinsic parameters from camera info
        width = cameraInfo.width
        height = cameraInfo.height
        ppx = cameraInfo.K[2]
        ppy = cameraInfo.K[5]
        fx = cameraInfo.K[0]
        fy = cameraInfo.K[4]    
        coeffs = [i for i in cameraInfo.D]

        if (any(np.array(pixel) > np.array([width, height]))):
            raise IndexError("Pixel is out of bounds from image: {}, {}".format(np.array(pixel), np.array([width, height])))
            
        x = (pixel[0] - ppx) / fx
        y = (pixel[1] - ppy) / fy 

        xo = x
        yo = y

        # BROWN CONRADY DISTORTION
        if cameraInfo.distortion_model == 'plumb_bob':
            for i in range(10):
                r2 = x*x + y*y
                
                icdist = 1 / (1+((coeffs[4]*r2 + coeffs[1])*r2 + coeffs[0])*r2)
                delta_x = 2 * coeffs[2] * x*y + coeffs[3] * (r2 + 2 * x*x)
                delta_y = 2 * coeffs[3] * x*y + coeffs[2] * (r2 + 2 * y*y)
                x = (xo - delta_x)*icdist
                y = (yo - delta_y)*icdist

        # KANNALA BRANDT4 DISTORTION
        elif cameraInfo.distortion_model == 'equidistant':
            rd = np.sqrt(x*x + y*y)
            if (rd < sys.float_info.epsilon):
                rd = sys.float_info.epsilon
            theta = rd
            theta2 = rd*rd

            for i in range(4):
                f = theta*(1 + theta2*(coeffs[0] + theta2*(coeffs[1] + theta2*(coeffs[2] + theta2*coeffs[3])))) - rd
                if (abs(f) < sys.float_info.epsilon): # already converged        
                    break
                    
                df = 1 + theta2*(3 * coeffs[0] + theta2*(5 * coeffs[1] + theta2*(7 * coeffs[2] + 9 * theta2*coeffs[3])))
                theta -= f / df
                theta2 = theta*theta
            
            r = np.tan(theta)
            x *= r / rd
            y *= r / rd

        return depth*np.array([x,y,1])

def main():

    rospy.init_node('sort_tracking')

    st = SortTrackingNode()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("error")
        pass
    print("Exiting process...")