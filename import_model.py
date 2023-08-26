import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import onnxruntime as ort
from time import time
import numpy as np

if __name__ == '__main__':
    
    jit_model = torch.jit.load('animesr.pt')
    jit_model.eval()

    image1 = Image.open('naruto.jpg')
    ow, oh = image1.size
    # image1 = fn.resize(image1, (320,320))
    image1 = transforms.ToTensor()(image1)

    # start = time()
    # output: torch.Tensor = jit_model(image1)
    # now = time()
    # print(round((now-start), 3))
    
    # image_ouput = transforms.ToPILImage()(output)
    # image_ouput = fn.resize(image_ouput, (oh*4,ow*4))

    # image_ouput.save('naruto_output.jpg')

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    sess = ort.InferenceSession('animesr.onnx', 
                                providers=['CPUExecutionProvider'],
                                # sess_options=sess_options
                                )
    input_name = sess.get_inputs()[0].name

    image1 = transforms.Resize((320,320))(image1)
    start = time()
    outputs = sess.run(None, {input_name: image1.numpy()})
    output_tensor = outputs[0]
    now = time()
    print(round((now-start), 3))

    output = transforms.ToPILImage()(torch.from_numpy(output_tensor))
    output = transforms.Resize((oh*4,ow*4))(output)
    output.save('onnx_output.jpg')


