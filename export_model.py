import sys
import torch
from PIL import Image
import coremltools as ct
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.append('sr')
from sr.animesr.archs.vsr_arch import MSRSWVSR
from sr.scripts.inference_animesr_frames import *

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

    image1 = Image.open('rose_32.jpg')
    image1 = transforms.ToTensor()(image1)

    # output = model(image1)
    # output = transforms.ToPILImage()(output)
    # output.save('output.jpg')

    example_input = torch.rand(3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    # scripted_model = torch.jit.script(traced_model)
    # optimized_model = optimize_for_mobile(scripted_model)
    # optimized_model.save('animesr.pt')

    torch.onnx.export(traced_model,
        example_input,             # model input (or a tuple for multiple inputs)
        "animesr.onnx",            # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )

    input_shape = ct.Shape(shape=(3,
        ct.RangeDim(lower_bound=64, upper_bound=1024, default=512),
        ct.RangeDim(lower_bound=64, upper_bound=1024, default=512)
    ))

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

