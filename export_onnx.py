import torch
from main_animesr import AnimeSR

if __name__ == '__main__':
    device = torch.device('cpu')

    model = AnimeSR(netscale=4)
   
    model_path = 'sr/weights/AnimeSR_v2.pth'
   
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet, strict=True)

    model.train(False)
    model.cpu().eval()
    
    rand_img = torch.rand(1, 3, 240, 240)

    traced_model = torch.jit.trace(model, rand_img)

    with torch.no_grad():
        scripted_model = torch.jit.script(traced_model)
        
        # optimized_model = optimize_for_mobile(scripted_model)
        # optimized_model.save('realesr_net.pt')
        
        torch.onnx.export(scripted_model,
            rand_img,
            "animesr.onnx",
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {'input': {2:'width', 3:'height'}, 'output':{2:'width', 3:'height'}}, 
            opset_version = 16,
        )