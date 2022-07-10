#! /usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import cv2 as cv

def main():
    rospy.init_node('image_sub')
    pub_sub = PubSub()
    pub_sub.start()
    rospy.spin()


class PubSub(object):

    def __init__(self):
        self.image = None
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(1)
        self.pub = rospy.Publisher('/resultado', Image, queue_size=3)

        rospy.Subscriber("/usb_cam/image_raw/", Image, self.callback)

    def callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.sleep(1)
        except e:
            print(e)
        else:
            # Lectura de la imagen
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8") # trainImage
            img2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
            # Conversión a grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #Inversion de imagen - para defectos de color claros - macilla
            inverted_image = np.invert(gray)

            # Processamiento de imagen ( smoothing )
            # Averaging
            blur = cv2.blur(inverted_image,(3,3))

            # Apply logarithmic transform
            img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

            # Specify the data type
            img_log = np.array(img_log,dtype=np.uint8)

            # Image smoothing: bilateral filter
            bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

            # Canny Edge Detection
            edges = cv2.Canny(bilateral,30,70)

            # Morphological Closing Operator
            kernel = np.ones((5,5),np.uint8)
            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Create feature detecting method
            #sift = cv2.xfeatures2d.SIFT_create()
            #surf = cv2.xfeatures2d.SURF_create()
            orb = cv2.ORB_create(nfeatures=1500)

            # Make featured Image
            keypoints, descriptors = orb.detectAndCompute(closing, None)
            featuredImg = cv2.drawKeypoints(closing, keypoints, None)

            self.image = closing

    def start(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                rospy.loginfo('Publicando Imagen')
                self.pub.publish(self.bridge.cv2_to_imgmsg(self.image))
            self.loop_rate.sleep()


if __name__ == '__main__':
    main()
