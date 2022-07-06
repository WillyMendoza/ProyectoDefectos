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
            # Cargamos la imagen
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8") # trainImage
            img2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            # Convertimos a escala de grises
            gris = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
            # Aplicar suavizado Gaussiano
            gauss = cv2.GaussianBlur(gris, (5,5), 0)
            
            # Detectamos los bordes con Canny
            canny = cv2.Canny(gauss, 50, 150)

            # Buscamos los contornos
            (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            # Mostramos la cantidad de objetos encontrados por consola
            print("He encontrado {} objetos".format(len(contornos)))
 
            cv2.drawContours(img,contornos,-1,(0,0,255), 2)

            self.image = canny

    def start(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                rospy.loginfo('Publicando Imagen')
                self.pub.publish(self.bridge.cv2_to_imgmsg(self.image))
            self.loop_rate.sleep()


if __name__ == '__main__':
    main()