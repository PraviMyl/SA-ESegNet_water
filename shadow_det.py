import cv2 as cv2
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import os, sys

path = './Dataset_II/train/'
dirs = os.listdir( path )

def shadow_det():
    for item in dirs:
        if os.path.isfile(path+item):
            rgb = io.imread(path+item)
            lab = color.rgb2lab(rgb)
            hsvImage = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            # plt.figure(dpi= 100)
            # plt.subplot(1,2,1),plt.title('rgb'), plt.imshow(rgb.astype('uint8')), plt.axis('off')
            # plt.subplot(1,2,2),plt.title('hsv'), plt.imshow(hsvImage.astype('uint8')), plt.axis('off')
            # plt.show()


            image_b = np.copy(lab[:, :, 2])
            image_a = np.copy(lab[:, :, 1])
            image_l = np.copy(lab[:, :, 0])
            # plt.figure(dpi= 150)
            # plt.axis('off')
            # plt.subplot(1,3,1),plt.title('l'), plt.imshow(image_l.astype('uint8')), plt.axis('off')
            # plt.subplot(1,3,2),plt.title('a'), plt.imshow(image_a.astype('uint8')), plt.axis('off')
            # plt.subplot(1,3,3),plt.title('b'), plt.imshow(image_b.astype('uint8')), plt.axis('off')
            # plt.show()


            lm=np.mean(lab[:,:,0], axis=(0, 1))
            am=np.mean(lab[:,:,1], axis=(0, 1))
            bm=np.mean(lab[:,:,2], axis=(0, 1))

            l_std = np.std(image_l)

            h = np.copy(hsvImage[:, :, 0])
            s = np.copy(hsvImage[:, :, 1])
            v = np.copy(hsvImage[:, :, 2])
            plt.figure(dpi= 150)
            # plt.axis('off')
            # plt.subplot(1,3,1),plt.title('h'), plt.imshow(h.astype('uint8')), plt.axis('off')
            # plt.subplot(1,3,2),plt.title('s'), plt.imshow(s.astype('uint8')), plt.axis('off')
            # plt.subplot(1,3,3),plt.title('v'), plt.imshow(v.astype('uint8')), plt.axis('off')
            # plt.show()

            mask = np.zeros((rgb.shape[0], rgb.shape[1], 3))

            if (am+bm)<=256:
                mask[(image_l <=(lm - l_std/3))] = 1
            else:
                mask[(image_l+image_b >= 25).all() and (image_l+image_b <=43).all()] = 1


            mask[(h >= 35).all() and (h <= 170).all() and (s <= 60).all() and (v <= 50)] = 1


            # plt.imshow((mask*255).astype(np.uint8)), plt.axis('off')
            plt.figure(dpi= 100)
            plt.subplot(1,2,1),plt.title('rgb'), plt.imshow(rgb.astype('uint8')), plt.axis('off')
            plt.subplot(1,2,2),plt.title('shadow detect'), plt.imshow(mask), plt.axis('off')
            plt.show()
            cv2.imwrite(os.path.join('./Dataset_II/train_shadow/',os.path.basename(path+item)), mask.astype('uint8')*255)
            # plt.imshow(mask), plt.axis('off')
            # plt.title('Shadow detected Image')
            # plt.show()

shadow_det()



