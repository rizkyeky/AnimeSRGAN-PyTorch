import torch
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort

if __name__ == '__main__':
    
    jit_model = torch.jit.load('animesr.pt')
    jit_model.eval()

    image1 = Image.open('rose_32.jpg')
    image1 = transforms.ToTensor()(image1)

    output = jit_model(image1)
    output = transforms.ToPILImage()(output)
    output.save('jit_output.jpg')

    sess = ort.InferenceSession('animesr.onnx')
    input_name = sess.get_inputs()[0].name

    image1 = transforms.Resize((480,480))(image1)
    outputs = sess.run(None, {input_name: image1.numpy()})
    output_tensor = outputs[0]
    print(output_tensor.shape)

    output = transforms.ToPILImage()(torch.from_numpy(output_tensor))
    output.save('onnx_output.jpg')


