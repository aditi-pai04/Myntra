import torch
from torchvision import transforms
from PIL import Image
import os
from networks import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint
import numpy as np

seg_path='C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/VITON-HD/checkpoints/seg_final.pth'
gmm_path='C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/VITON-HD/checkpoints/gmm_final.pth'
alias_path='C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/VITON-HD/checkpoints/alias_final.pth'
class Options:
    def __init__(self):
        self.semantic_nc = 13  # Typical value for human parsing
        self.init_type = 'xavier'
        self.init_variance = 0.02
        self.grid_size = 5
        self.norm_G = 'spectralaliasinstance'
        self.ngf = 64
        self.num_upsampling_layers = 'most'
        self.load_height = 1024
        self.load_width = 768

opt = Options()

def load_model(model_path):

    print(f"Loading checkpoints:\nSeg: {seg_path}\nGMM: {gmm_path}\nAlias: {alias_path}")

    if not os.path.exists(seg_path) or not os.path.exists(gmm_path) or not os.path.exists(alias_path):
        raise ValueError(f"Checkpoint paths are not valid: {seg_path}, {gmm_path}, {alias_path}")
    # Updating based on typical values
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, seg_path)
    load_checkpoint(gmm, gmm_path)
    load_checkpoint(alias, alias_path)

    seg.eval()
    gmm.eval()
    alias.eval()
    return seg, gmm, alias

def perform_try_on(seg, gmm, alias, user_image_path, clothes_image_path, output_image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
    ])

    # Load and transform user image
    image = Image.open(user_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load and transform clothes image
    clothes = Image.open(clothes_image_path).convert('RGB')
    clothes_tensor = transform(clothes).unsqueeze(0)  # Add batch dimension
    
    # # Pad tensors to match required dimensions
    # input_pad = torch.zeros_like(image_tensor)
    # clothes_pad = torch.zeros_like(clothes_tensor)
    print(type(clothes_tensor))
    # Concatenate tensors with padding
    input_tensor = torch.cat([image_tensor, torch.zeros(1, 4, 256, 192)], dim=1)
    # Assuming adjustment to match 7 channels
    clothes_tensor = torch.cat([clothes_tensor, torch.zeros(1, 7, 256, 192)], dim=1)

    
    with torch.no_grad():
        parse_pred_down = seg(input_tensor)
        parse_pred = parse_pred_down.argmax(dim=1)[:, None]

        gmm_input = torch.cat((parse_pred.float(), input_tensor), dim=1)
        _, warped_grid = gmm(gmm_input, clothes_tensor)
        warped_clothes = torch.nn.functional.grid_sample(clothes_tensor, warped_grid, padding_mode='border')

        output = alias(torch.cat((input_tensor, warped_clothes), dim=1))

    output_image = transforms.ToPILImage()(output.squeeze().cpu())
    output_image.save(output_image_path)
    print(f"Output image saved to {output_image_path}")