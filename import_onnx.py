import onnx
import torch
import onnxruntime
import numpy as np
import cv2

def open_with_opencv():
    net = cv2.dnn.readNetFromONNX('animesr_320_nobatch.onnx')

def open_with_onnxruntime():
    session = onnxruntime.InferenceSession('/Users/eky/Projects/_pretrained/animesr.onnx')

    ori_img = cv2.imread('imgs/naruto.jpg')
    ori_h, ori_w, _ = ori_img.shape
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    img = (np.array(img) / 255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_feed = {input_name: img}
    output = session.run([output_name], input_feed)

    output = output[0].clip(0, 1) * 255
    output = output.astype(np.uint8)
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = cv2.resize(output, (ori_w*4, ori_h*4))
    # cv2.imwrite('naruto_animesr.jpg', output)

    ori_img = cv2.resize(ori_img, (ori_w*2, ori_h*2))
    output = cv2.resize(output, (ori_w*2, ori_h*2))
    stack = np.column_stack((ori_img, output))

    cv2.imshow('output', stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    open_with_onnxruntime()
    # open_with_opencv()
