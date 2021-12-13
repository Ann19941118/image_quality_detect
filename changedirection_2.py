'''
Descripttion: 
version: 1.0
Author: Ge Jun
Date: 2021-07-06 22:39:56
LastEditors: Ge Jun
LastEditTime: 2021-08-16 16:15:46
'''
import os
import cv2
import cv2
import numpy as np
from PIL import Image
# import base64

def detect_rotation(img_name,img):

    rotation = 0
    rota_split = img_name.split('-')
    if 'Copy' in rota_split[-1]:
        rota_id = rota_split[-2][0]
    else:
        rota_id = rota_split[-1][0]
    if rota_id in ['3','2']:
            rotation = -90
        #下>上
    elif  rota_id in ['1','4']:
            rotation = 90
  
      #旋转图像
    image = Image.fromarray(img)
    im_rotate = image.rotate(rotation)
    im_rotate = np.array(im_rotate)
    return im_rotate
 
if __name__ == "__main__":
    json_floder_path = './normal'

    json_names = os.listdir(json_floder_path)
    for jpg_name in json_names:
        if jpg_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(json_floder_path,jpg_name))


            im_rotate = detect_rotation(jpg_name,image)
            cv2.imwrite('./nor_rotate2/'+jpg_name, im_rotate)
        # cv2.imshow("image1",image )
        # cv2.imshow("image2",im_rotate)
        # cv2.waitKey(0)