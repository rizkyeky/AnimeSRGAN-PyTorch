import sys
import torch
from PIL import Image
# import coremltools as ct
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.append('sr')
from sr.animesr.archs.vsr_arch import MSRSWVSR
from sr.scripts.inference_animesr_frames import *

def roundToNearestMultipleOf10(num: int):
    remainder = num % 10
    return num - remainder

def findNearestRatioOptions(first: int, second: int):
  
    index = 0

    targetRatio = first / second
    aspectRatios = [
        1.0, # 1
        4.0/3.0, # 1,3333
        3.0/2.0, # 1,5
        5.0/3.0, # 1,666
        16.0/9.0 # 1,777
    ]
  
    minDifference = abs(targetRatio - aspectRatios[0])
    for i in range(5):
        distance = abs(targetRatio - aspectRatios[i])
        if (distance < minDifference):
            minDifference = distance
            index = i

    return index

def findPerfectSize(width: int, height: int) -> tuple:
    width = roundToNearestMultipleOf10(width)
    height = roundToNearestMultipleOf10(height)
    ratioOptions = [
        (1,1), # 1
        (4,3), # 1,3333
        (3,2), # 1,5
        (5,3), # 1,6
        (16,9) # 1,777
    ]
    newRatio = 1

    if (width > height):
        # Landscape
        newIndex = findNearestRatioOptions(width, height)
        newRatio = ratioOptions[newIndex]
        height = int((width / newRatio[0]) * newRatio[1])
        height = height if height % 2 == 0 else height+1
        height += 2
        height = height+2 if height % 10 == 0 else height+4
    elif (width < height):
        # Portrait
        newIndex = findNearestRatioOptions(height, width)
        newRatio = ratioOptions[newIndex]
        width = int((height / newRatio[0]) * newRatio[1])
        width = width if width % 2 == 0 else width+1

    return (int(width), int(height))

class AnimeSR(MSRSWVSR):
    def __init__(self, netscale):
        super(AnimeSR, self).__init__(64, [5, 3, 2], netscale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        b, c, h, w = x.size()
        state = x.new_zeros(1, 64, h, w)
        out = x.new_zeros(1, c, h * self.netscale, w * self.netscale)
        stack = torch.cat((x, x, x), dim=1)
        out, state = self.cell(stack, out, state)
        return out.squeeze(0)

if __name__ == '__main__':

    netscale = 4
    # mod_scale = 4
    # input_rescaling_factor = 1.0
    device = torch.device('cpu')
   
    model = AnimeSR(netscale=netscale)
   
    model_path = 'sr/weights/AnimeSR_v2.pth'
   
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet, strict=True)
    model.eval()
    model = model.to(device)

    # image1 = Image.open('naruto.jpg')
    # ow, oh = image1.size
    # image1 = transforms.Resize((320,320))(image1)
    # image1 = transforms.ToTensor()(image1)

    # output = model(image1)
    # output = transforms.ToPILImage()(output)
    # output = transforms.Resize((oh*2,ow*2))(output)
    # output.save('naruto1_output1.jpg')

    example_input = torch.rand(3, 320, 320)
    traced_model = torch.jit.trace(model, example_input)

    scripted_model = torch.jit.script(traced_model)
    # optimized_model = optimize_for_mobile(scripted_model)
    # optimized_model.save('animesr.pt')

    torch.onnx.export(scripted_model,
        example_input,
        "animesr.onnx",
        input_names = ['input'],
        output_names = ['output'],
        # dynamic_axes = {'input': {1:'width', 2:'height'}, 'output':{1:'width', 2:'height'}}, 
        opset_version = 16,
        # operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN
    )

    # input_shape = ct.Shape(shape=(3,
    #     ct.RangeDim(lower_bound=64, upper_bound=1024, default=512),
    #     ct.RangeDim(lower_bound=64, upper_bound=1024, default=512)
    # ))

    # ct_model = ct.convert(traced_model,
    #     inputs=[ct.ImageType(
    #         shape=input_shape,
    #         scale=1/255,
    #         color_layout=ct.colorlayout.RGB,
    #         # name="input"
    #     )],
    #     outputs=[ct.ImageType(
    #         color_layout=ct.colorlayout.RGB
    #     )],
    #     # compute_units=ct.ComputeUnit.ALL,
    #     # minimum_deployment_target=ct.target.iOS16,
    #     # compute_precision=ct.precision.FLOAT32,
    #     # convert_to="mlprogram",
    # )

    # ct_model.save('animesr.mlmodel')
   
    # image_input = ct.ImageType(shape=(1, 224, 224, 3,),
    #                         bias=[-1,-1,-1], scale=1/127)
   
    # classifier_config = ct.ClassifierConfig(class_labels)
   
    # model = ct.convert(
    #     model, 
    #     convert_to="mlprogram",
    #     # inputs=[image_input], 
    #     # classifier_config=classifier_config,
    # )

