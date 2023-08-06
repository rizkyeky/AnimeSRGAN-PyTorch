import torch
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
from time import time

if __name__ == '__main__':
    
    jit_model = torch.jit.load('animesr.pt')
    jit_model.eval()

    image1 = Image.open('rose_32.jpg')
    image1 = transforms.ToTensor()(image1)

    start = time()
    output = jit_model(image1)
    now = time()
    print(round((now-start), 3))
    # output = transforms.ToPILImage()(output)
    # output.save('jit_output.jpg')

    # sess_options = ort.SessionOptions()
    # sess_options.log_severity_level = 0
    sess = ort.InferenceSession('animesr.onnx', 
                                providers=['CPUExecutionProvider'],
                                # sess_options=sess_options
                                )
    input_name = sess.get_inputs()[0].name

    image1 = transforms.Resize((224,224))(image1)
    start = time()
    outputs = sess.run(None, {input_name: image1.numpy()})
    now = time()
    print(round((now-start), 3))
    output_tensor = outputs[0]
    # print(output_tensor.shape)

    # output = transforms.ToPILImage()(torch.from_numpy(output_tensor))
    # output.save('onnx_output.jpg')


