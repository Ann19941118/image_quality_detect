import cv2
import numpy as np
import os, json
import matplotlib.pyplot as plt
import base64

def load_image(path,RGB=True,gray=False):
    gray_flag = 0 if gray else 1
    img = cv2.imread(path, gray_flag)
    if RGB:img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  



class image_generator_landmarksAware:
            
    """Generate tensors of images for data augmentation.
      The data will be looped over.
      
      # Arguments
        image: image like Numpy array 
        keypoints: list of coordinate points ex. [x1, y1, x2, y2, ...]
        rotate_range: Int. or tuple of Int(s) Degree range for random rotations.
        shift_range: Float, or tuple of Floats
            - float: fraction of total width or height

        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.

        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, `[lower, upper] = [abs(1-zoom_range), zoom_range]`.
            
        BorderMode: One of {0, 1, 2 or 3}.
            Default is 1 -> 'nearest'.
            Points outside the boundaries of the input are filled
            according to the given mode:
            - 0 -> 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 1 -> 'nearest':  aaaaaaaa|abcd|dddddddd
            - 2 -> 'reflect':  abcddcba|abcd|dcbaabcd
            - 3 -> 'wrap':  abcdabcd|abcd|abcdabcd

        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        

        preprocessing_function: function that will be applied on each input.
            The function will run before the image is resized and augmented.
            The function should take one argument:
            one image (NumPy tensor with rank 3),
            and should output a NumPy tensor with the same shape.
            
    # Examples:
        comming Soon with v0.4 release 
     
     
     
     """
    def __init__(self,
                 image,
                 keypoints,
                 rotate_range=0.0,
                 shift_range=0.0,     
                 blur_range=0,
                 noise_range=0.0,
                 zoom_range=0.0,
                 brightness_range=None,
                 sharpen=False,
                 horizontal_flip=False,
                 vertical_flip=False,
                 preprocessing_function=None,
                 target_shape=None,
                 p_rotate=0.2,
                 p_shift=0.2,
                 p_bright=0.2,
                 p_blur=0.1,
                 p_noise=0.1,
                 p_zoom=0.2,
                 p_vflip=0.1,
                 p_hflip=0.1,
                 p_sharpen=0.1,
                 epochs=1,
                 BorderMode=1,
                 informative=False,
                 shutdown_Warrings=True,
                 allow_repeate=False):
        

        self.image = image
        self.img = image.copy() / 255.0
        self.keypoints = keypoints
        self.k = np.array(self.keypoints)
        self.blur_range = blur_range
        self.noise_range = noise_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.sharpen = sharpen
        self.preprocessing_function = preprocessing_function
        
        self.p_rotate = p_rotate
        self.p_shift = p_shift
        self.p_bright = p_bright
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_zoom = p_zoom
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_sharpen = p_sharpen
        self.epochs = epochs
        self.allow_repeate = allow_repeate
        
        self.informative = informative
        self.BorderMode = BorderMode
        self.h, self.w  = self.image.shape[:2]
        self.center = (self.w//2, self.h//2)
        self.target_shape = target_shape
        
        # ---------Checking for valid ranges-------------
        if isinstance(rotate_range, (int,float)):
            self.rotate_range = [-rotate_range, rotate_range]
        elif isinstance(rotate_range,(list,tuple)):
            self.rotate_range = rotate_range
        else:
            raise ValueError('rotate_range should be float tuple or list of two floats.'
                            'Received: %s' % (rotate_range,))
            
        if isinstance(shift_range, (int,float)):
            self.shift_range = [-shift_range, shift_range]
        elif isinstance(shift_range,(list,tuple)):
            self.rotate_range = rotate_range
        else:
            raise ValueError('shift_range should be float tuple or list of two floats.'
                            'Received: %s' % (shift_range,))
                       
        if brightness_range is not None:
            if (not isinstance(brightness_range, (tuple, list)) or len(brightness_range) != 2):
                raise ValueError('`brightness_range should be tuple or list of two floats. '
                                 'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range
            
        if isinstance(zoom_range, (int,float)):
            self.zoom_range = [abs(1-zoom_range), zoom_range]         
        elif isinstance(zoom_range, (list,tuple)):
            self.zoom_range = zoom_range
        else:
            raise ValueError('zoom_range should be float tuple or list of two floats.'
                            'Received: %s' % (zoom_range,))
        
        
        
    def generate(self, num_images, batch_size=1):   
        self.augmentation_list = [self._rotate, self._shift,
                                  self._blur,self._apply_noise, self._scale_image]
        if self.brightness_range:self.augmentation_list.append(self._brighten)
        if self.horizontal_flip:self.augmentation_list.append(self._hflip)
        if self.vertical_flip:self.augmentation_list.append(self._vflip)
        if self.sharpen:self.augmentation_list.append(self._sharpen)
        if self.preprocessing_function:
            input_image, input_keypoints = self.preprocessing_function(self.img, self.k)
        else:input_image, input_keypoints = self.img, self.k
        images_counter = 0
        while images_counter != num_images:
            #! implement batches comming Here
            output_image, output_keypoints = input_image, input_keypoints
            for augFunc in np.random.choice(self.augmentation_list, size=self.epochs, replace=self.allow_repeate, p=self.__stat_call_with_probability()):    
                output_image, output_keypoints = augFunc(output_image, output_keypoints)
            
            output_image, output_keypoints = self._resize(output_image, output_keypoints)
            output_image = self.__perfect_normlize(output_image)
            images_counter += 1 
            yield output_image, output_keypoints
        
        
    def _rotate(self, img, keypoints):
        angle = np.random.randint(*self.rotate_range)
        if self.informative:print(f'rotate has been called with {angle} degree')
        radian_angle = (-angle * np.pi) / 180.
        M = cv2.getRotationMatrix2D(self.center, angle, 1)
        rotated_img = cv2.warpAffine(img, M, (self.w,self.h),borderMode=self.BorderMode,flags=cv2.INTER_CUBIC)
        # keypoints augmention process
        keypts = keypoints - self.center[0] 
        keypts = np.array([keypts[0::2]*np.cos(radian_angle) - keypts[1::2]*np.sin(radian_angle),
                          keypts[0::2]*np.sin(radian_angle) + keypts[1::2]*np.cos(radian_angle)])
        keypts += self.center[0]
        keypts = np.array([(x,y) for x,y in zip(keypts[0], keypts[1])])
        if np.any(keypts<0):
            print('[Warning] the image ratation has not been done cuz we lost some keypoints try with less ratation range')
            return img, keypoints # or return None
            # return img,None
        return rotated_img, keypts.flatten()
        
    def _shift(self, img, keypoints):  
        if self.informative:print('shift has been called')
        x_shift, y_shift = np.random.uniform(*self.shift_range,size=2)
        x_shift, y_shift = int(self.w * x_shift) ,int(self.h * y_shift)
        M = np.float32([[1,0,x_shift],[0,1,y_shift]])
        shifted_img = cv2.warpAffine(img, M, (self.w,self.h),borderMode=self.BorderMode)
        # keypoints augmentations process
        keypts = keypoints.copy()
        keypts[::2] = keypts[::2] +  x_shift # the shift in the x-axis
        keypts[1::2] = keypts[1::2] + y_shift
        if np.any(keypts<0):
            print('[Warning] the image shifting has not been done cuz we lost some keypoints try with less shift range')
            return img, keypoints # or return None
            # return img,None
        return shifted_img, keypts
    
    def _brighten(self, img, keypoints):
        if self.informative:print('brighten has been called')
        brightness_range = np.random.uniform(*self.brightness_range)
        brighten_img = img * brightness_range
        return brighten_img, keypoints
    
    def _blur(self, img, keypoints):
        if self.blur_range==0:return img, keypoints
        k = np.random.choice(self.blur_range)
        if self.informative:print(f'blur has been applied wiith value {k}')
        kernel=(k, k)
        blured_img = cv2.blur(img, kernel)
        return blured_img, keypoints
    
    def _apply_noise(self, img, keypoints):
        if self.informative:print('noise has been applied')
        if self.noise_range ==0:return img, keypoints
        noisy_image = cv2.add(img, self.noise_range * np.random.randn(*self.image.shape))
        return noisy_image , keypoints
    
    def _scale_image(self, img, keypoints): 
        if self.informative:print('zoom has been applied')
        ratio = np.random.uniform(*self.zoom_range)
        center_Shift = (1-ratio) * self.center[0]
        M = np.float32([[ratio , 0 , center_Shift],[0, ratio , center_Shift]])
        scaled_img = cv2.warpAffine(img, M, (self.w, self.h),borderMode=self.BorderMode)
        #keypoints augmentaion
        keypts = keypoints.copy()
        keypts[0::2] =  keypts[0::2] * ratio + center_Shift
        keypts[1::2] = keypts[1::2] * ratio + center_Shift
        if np.any(keypts<0):
            print('[Warning] the image zoom has not been done cuz we lost some keypoints try with less zoom range')
            return img, keypoints 
            # return img,None
        return scaled_img, keypts
    
    def _hflip(self, img, keypoints):
        if self.informative:print('applied horizontal flip')
        keypts= keypoints - self.center[0]        
        M = np.float32([[-1, 0, self.w-1], [0, 1, 0]])
        keypts[::2] = - keypts[::2]
        fliped_img = cv2.warpAffine(img, M, (self.w, self.h))
        keypts += self.center[0]
        return fliped_img, keypts

    def _vflip(self, img, keypoints):
        if self.informative:print('applied vertical flip')
        keypts= keypoints - self.center[0]        
        M = np.float32([[1, 0, 0], [0, -1, self.h - 1]])
        keypts[1::2] =  -keypts[1::2]
        fliped_img = cv2.warpAffine(img, M, (self.w, self.h))
        keypts += self.center[0]
        return fliped_img, keypts
    
    def _sharpen(self, img, keypoints):
        if self.informative:print('sharpen has been applied')
        kernel = np.array([[0, -0.5, 0], [-0.5, 3 , -0.5], [0, -0.5, 0]])
        sharpen_img = cv2.filter2D(img, -1, kernel)
        return sharpen_img, keypoints
    
    def _resize(self, img, keypoints):
        if self.target_shape is None: return img, keypoints
        orignal_shape = self.image.shape
        resized_img = cv2.resize(img, self.target_shape[:2]) 
        keypts = keypoints.copy()
        keypts[::2] = keypts[::2] * self.target_shape[1] / float(orignal_shape[1])
        keypts[1::2] =keypts[1::2] * self.target_shape[0] / float(orignal_shape[0])
        return resized_img, keypts
              
    def __stat_call_with_probability(self):
        a=np.random.dirichlet(np.ones(len(self.augmentation_list)), size=1).flatten()
        values = [self.p_rotate, self.p_shift, 
                  self.p_blur,self.p_noise, self.p_zoom]
        if self.brightness_range is not None :values.append(self.p_bright)
        if self.horizontal_flip:values.append(self.p_hflip)
        if self.vertical_flip:values.append(self.p_vflip)
        if self.sharpen:values.append(self.p_sharpen)
        for idx in range(len(values)): a[idx] = values[idx]
        a /= a.sum()
        return a
        
    def __perfect_normlize(self,image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
 
    def __str__(self):
        return (f''' the rotate range parameter provided were {self.rotate_range} ,
the shift_range parameter provided were {self.shift_range},
the zoom_range  parameter provided were {self.zoom_range},
the brightness_range parameter provided were {self.brightness_range},
the blur_range parameter provided were {self.blur_range},
the noise_range parameter provided were {self.noise_range},
the sharpen parameter is set to {self.sharpen},
horizontal_flip is set to {self.horizontal_flip},
vertical_flip is set to {self.vertical_flip},
the output image should have the {self.image.shape if self.target_shape is None else self.target_shape} shape''')
    
        
      

def display_with_landmark(image,keypoints,color_codes=None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image)
    keypts = [(x,y) for x,y in zip(keypoints[::2],keypoints[1::2])]
    for point in keypts:
        ax.scatter(*point, c='r')
    plt.show()

def txt_path(jpg_name):
    root_path = '/home/think/文档/yolov5/runs/detect'
    file_folders = ['exp6']
    txt_path1 = ''
    for folder in file_folders:
        path_tmp = os.path.join(root_path,folder,'labels')
        txt_path1 = os.path.join(path_tmp,jpg_name).replace('jpg','txt')
        if os.path.isfile(txt_path1):
            return txt_path1

def decode_txt(txt_path):
    if os.path.isfile(txt_path):
        with open(txt_path,'r') as f:
            line_ = f.readline()
        cxywh = [round(float(x)*1600,4) for x in line_.split(' ')[1:5]]

        xcenter,ycenter,w,h =  cxywh[0],cxywh[1], cxywh[2],cxywh[3]
        # xmin,ymin,w0,h0 = round(xcenter-w/2),round(ycenter-h/2),round(w),round(h)
        xmin,ymin,xmax,ymax = round(xcenter-w/2)-5,round(ycenter-h/2)-5,round(xcenter+w/2)+5,round(ycenter+h/2)+5

        pts =[xmin,ymin,xmax,ymin,xmin,ymax,xmax,ymax]
    else:
        pts = []
    return pts

# # class MyEncoder(json.JSONEncoder):
# #     def default(self, obj):
#         if isinstance(obj, bytes):
#             return str(obj, encoding='utf-8');
#         return json.JSONEncoder.default(self, obj)
if __name__ == "__main__":

    file_path = './nor_rotate2'
    jpg_names = os.listdir(file_path)
    leng = len(jpg_names)
    for i,jpg_name in enumerate(jpg_names):
        if jpg_name.endswith('.jpg'):
            print('%s/%s: %s'%(i,leng,jpg_name))
            img = load_image(os.path.join(file_path,jpg_name))
            txt_path1 = txt_path(jpg_name)
            # print(txt_path1)
            if txt_path1:
                points = decode_txt(txt_path1)
                gen = image_generator_landmarksAware(image=img,
                                                    keypoints=points,
                                                    rotate_range=(-5,5),
                                                    shift_range=0.10,
                                                    brightness_range=(0.8,1.2),   
                                                    noise_range=0.00,
                                                    allow_repeate=True,
                                                    blur_range=(3,1),
                                                    horizontal_flip=True,
                                                    zoom_range=(0.9,1.1),
                                                    target_shape=(1600,1600,3),
                                                    sharpen=False,
                                                    epochs=10)

                # for image, kpoints in gen.generate(20):
                #     display_with_landmark(image, kpoints)

                for i, out in enumerate(gen.generate(25)):
                    image, k = out
                    image = image*255
                    # landmarks = [[int(k[i]),int(k[i+1])] for i in range(0,len(k),2)]
                    cv2.imwrite('./nor_aug2/'+jpg_name[:-4]+'_'+str(i)+'.jpg',image) # save the augmented image
                # data["imageData"]= string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()   #save the augmented json
                # data['shapes'][0]['points'] = landmarks
                # with open('./data/coco/img_aug/'+jpg_name[:-4]+'_'+str(i)+'.json', 'w',encoding='utf-8') as f:
                #     json.dump(data,f)

