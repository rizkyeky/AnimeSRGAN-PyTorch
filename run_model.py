import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import onnxruntime as ort
from time import time
import numpy as np
import cv2

if __name__ == '__main__':
    
    cv_model = cv2.dnn.readNetFromONNX('animesr.onnx')

    image1 = cv2.imread('naruto.jpg')
    oh, ow, c = image1.shape
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    input_blob = cv2.dnn.blobFromImage(image1, 1.0/255, (320,320))
    
    cv_model.setInput(input_blob)
    output = cv_model.forward()

    output *= 255
    output = output.transpose(1,2,0)

    brgImg = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    resizedImg = cv2.resize(brgImg, (ow*4,oh*4), interpolation=cv2.INTER_LINEAR_EXACT)

    cv2.imwrite('output_.jpg', resizedImg)


