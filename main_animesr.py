import sys
import torch
from PIL import Image
# import coremltools as ct
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.nn import functional as F

sys.path.append('sr')
from sr.animesr.archs.vsr_arch import MSRSWVSR
from sr.scripts.inference_animesr_frames import *

class AnimeSR(MSRSWVSR):
    def __init__(self, netscale):
        super(AnimeSR, self).__init__(64, [5, 3, 2], netscale)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.size()
        x = x.unsqueeze(0)
        state = x.new_zeros(1, 64, h, w)
        out = x.new_zeros(1, c, h * self.netscale, w * self.netscale)
        stack = torch.cat((x, x, x), dim=1)
        out, state = self.cell(stack, out, state)
        return out.squeeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        # print(x.shape)
        x, ori_h, ori_w = self.pre_process(x)
        # print(x.shape)
        x = self._forward(x)
        x = x.unsqueeze(0)
        x = F.interpolate(input=x, size=(ori_h*self.netscale, ori_w*self.netscale), mode="bilinear", align_corners=False)
        return x
    
    def pre_process(self, input: torch.Tensor) -> (torch.Tensor, int, int):
        output_stride = 32
        ori_height, ori_width = input.shape[1:]
        new_h = (ori_height // output_stride) * output_stride
        new_w = (ori_width // output_stride) * output_stride
        input = input.unsqueeze(0)
        output = F.interpolate(input=input, size=(new_h, new_w), mode="bilinear", align_corners=False)
        output = output.squeeze()
        return output, ori_height, ori_width

if __name__ == '__main__':

    device = torch.device('cpu')
   
    model = AnimeSR(netscale=4)
   
    model_path = '/Users/eky/Projects/_pretrained/animesr/AnimeSR_v2.pth'
   
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet, strict=True)

    # model = mobilenet_v3_large(weights='DEFAULT')

    model.train(False)
    model.cpu().eval()

    img = cv2.imread('imgs/rose.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320,320))
    img = (np.array(img) / 255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, 0)

    input_tensor = torch.from_numpy(img)

    output = model(input_tensor)
    
    output = output.cpu().detach().numpy()
    output = output.clip(0, 1) * 255
    output = output.astype(np.uint8)
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs/rose_animesr.jpg', output)

