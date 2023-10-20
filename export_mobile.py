import torch
import coremltools as ct
import numpy as np
import torchvision
from main_animesr import AnimeSR

from torch.backends._coreml.preprocess import (
    CompileSpec,
    TensorSpec,
    CoreMLComputeUnit,
)

def mobilenetv2_spec():
    return {
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 224, 224],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 1000],
                ),
            ),
            backend=CoreMLComputeUnit.ALL,
            allow_low_precision=True,
        ),
    }


def export_torch_coreml():
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    model = torch.jit.trace(model, example)
    compile_spec = mobilenetv2_spec()
    mlmodel = torch._C._jit_to_backend("coreml", model, compile_spec)
    mlmodel._save_for_lite_interpreter("mobilenetv2_coreml.ptl")

def export_mobilenet_v2_coreml():
    
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.train(False)
    model.cpu().eval()

    example_input = torch.rand(1, 3, 240, 240)
    traced_model = torch.jit.trace(model, example_input)

    with torch.no_grad():
        scripted_model = torch.jit.script(traced_model)

        input_shape = ct.Shape(shape=(1,3,
            ct.RangeDim(lower_bound=100, upper_bound=1000, default=320),
            ct.RangeDim(lower_bound=100, upper_bound=1000, default=320)
        ))

        model_ct = ct.convert(scripted_model,
            inputs=[ct.TensorType(shape=input_shape, name="input")],
            outputs=[ct.TensorType(name="output")],
            convert_to="mlprogram",
        )

        model_ct.save("mobilevitv2.mlpackage")

    # Test the model with predictions.
    input_1 = np.random.rand(1, 3, 640, 300)
    input_2 = np.random.rand(1, 3, 190, 266)

    output_1 = model_ct.predict({"input": input_1})["output"]
    print("output shape {} for input shape {}".format(output_1.shape, input_1.shape))
    output_2 = model_ct.predict({"input": input_2})["output"]
    print("output shape {} for input shape {}".format(output_2.shape, input_2.shape))

def export_animesr_coreml():
    
    model = AnimeSR(netscale=4)
   
    model_path = 'sr/weights/AnimeSR_v2.pth'
   
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet, strict=True)

    model.train(False)
    model.cpu().eval()

    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    with torch.no_grad():
        scripted_model = torch.jit.script(traced_model)

        input_shape = ct.Shape(shape=(1,3,
            ct.RangeDim(lower_bound=100, upper_bound=1000, default=320),
            ct.RangeDim(lower_bound=100, upper_bound=1000, default=320)
        ))

        model_ct = ct.convert(scripted_model,
            inputs=[ct.TensorType(shape=input_shape, name="input")],
            outputs=[ct.TensorType(name="output")],
            convert_to="mlprogram",
        )

        model_ct.save("animesr.mlpackage")

    # Test the model with predictions.
    input_1 = np.random.rand(1, 3, 640, 300)
    input_2 = np.random.rand(1, 3, 190, 266)

    output_1 = model_ct.predict({"input": input_1})["output"]
    print("output shape {} for input shape {}".format(output_1.shape, input_1.shape))
    output_2 = model_ct.predict({"input": input_2})["output"]
    print("output shape {} for input shape {}".format(output_2.shape, input_2.shape))

def export_from_onnx_to_coreml():
    model = ct.converters.onnx.convert(model='animesr.onnx')
    model.save("animesr.mlpackage")

if __name__ == "__main__":
    export_from_onnx_to_coreml()