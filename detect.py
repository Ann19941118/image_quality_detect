import torch
import numpy as np
from torchvision import transforms
from model import Model
import cv2

class Abnormal2:
    def __init__(self) -> None:
        #-----------------------类别---------------------------------
        self.classes = ['abnormal','normal']

        #--------------------载入模型---------------------------------
        self.model = Model(num_classes=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load('best.pth', map_location=device)
        self.model.load_state_dict(state_dict)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        if torch.cuda.is_available():
            self.model.cuda()

    def detect(self,image):
        src = image.copy()
        image =  self.transforms(image)
        image = np.expand_dims(image,0)
    
        print(image.shape)

        with torch.no_grad():
            image = torch.Tensor(image).cuda()
            out = self.model(image)
            pred = torch.max(out, 1)[1]
            id = pred.cpu().numpy()[0]
            cls = self.classes[pred.cpu().numpy()[0]]

            judge = 'NG' if id ==0 else 'OK'
            cv2.putText(src,judge+"   "+cls,(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

            return src, judge

if __name__ == '__main__':
    ab_model = Abnormal2()
    import glob
    filenames = glob.glob("/home/think/文档/images/xray_blur/*.jpg")
    print(filenames)
    for file in filenames:
        print(file)
        image = cv2.imread(file)
        r_image,cls = ab_model.detect(image)
        cv2.imshow('1',r_image)
        cv2.waitKey(1000)
