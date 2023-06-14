
# AnimeSuperResolution

## Original Repo
https://github.com/TencentARC/AnimeSR

** Nothing changes from original repo, just wrap the model for export to CoreML, TorchScript, and ONNX

## Depedencies

```
pip install torch torchvision
pip install basicsr
pip install coremltools
```

| Deployment | Success |
| ------ | ------ |
| TorchScript | ✅ |
| ONNX | ✅ |
| CoreML | ValueError: Core ML only supports tensors with rank <= 5. Layer "x_view", with type "reshape", outputs a rank 6 tensor |

