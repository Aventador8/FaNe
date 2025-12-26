import numpy as np
from einops import rearrange
import cv2
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from  torchvision import transforms

from codes.fane.data.pretrain.visual_mimiccxr import MimicCxrDataset
from codes.fane.encoders.image.resnet_encoder import ResEncoder
from codes.fane.encoders.language.bert import ClinicalBERT
from codes.fane.models.test_mcga import Mcga

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 1
image_encoder = ResEncoder(model_name="resnet_50", weights_path="xxxx.pth")
text_encoder = ClinicalBERT()

ckpt_path = "xxxx.ckpt"

model = Mcga.load_from_checkpoint(
    ckpt_path,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    down_stream=True,
    gpus=[0],
    strict=False
)

datamodule = MimicCxrDataset(
    dataset_path='data.json',
    num_colors=3,
    rate=1e-3,
    image_transform=[transforms.Resize(size=[224, 224])]
)

test_dataloader = DataLoader(datamodule, batch_size=batch_size, num_workers=16)

print("test_loader: {}".format(len(test_dataloader)))

def gaussian_blur_torch(atten_map, kernel_size=11, sigma=2):

    kernel = torch.exp(-torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, 1).to('cuda')
    kernel_h = kernel
    kernel_v = kernel.permute(0, 1, 3, 2)

    atten_map = atten_map.view(1, 1, 224, 224).to('cuda')

    atten_map = F.conv2d(atten_map, kernel_h, padding=(kernel_size // 2, 0))

    atten_map = F.conv2d(atten_map, kernel_v, padding=(0, kernel_size // 2))

    return atten_map.squeeze()


def apply_colormap_torch(atten_map, colormap='jet'):
    atten_map = atten_map / 255.0
    if colormap == 'jet':
        r = torch.clamp(torch.where(atten_map < 0.5, 0, (atten_map - 0.5) * 2), 0, 1)
        g = torch.clamp(torch.where(atten_map < 0.5, atten_map * 2, 1 - (atten_map - 0.5) * 2), 0, 1)
        b = torch.clamp(torch.where(atten_map > 0.5, 0, 1 - atten_map * 2), 0, 1)
    return torch.stack([r, g, b], dim=-1)

model = model.to('cuda')
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        image = batch['image'].to('cuda')
        text = batch['text'].to('cuda')
        sentences_list = batch['text_meta']['sentences_list']

        local_image_embed, global_image_embed = model.encode_image(image)
        local_text_embed = model.encode_text(text)
        local_text_embed_stacks = model.item_gather(local_text_embed, batch)
        global_text_embed = model.get_global_text_representation(local_text_embed_stacks)


        for idx in range(batch_size):
            local_text_embed = local_text_embed_stacks[idx]
            local_image_embed = local_image_embed[idx]
            text_to_local_image_embed, attention_probs, _ = model.local_cross_attention(
                local_image_embed, local_text_embed, sparse=True
            )

            num_sentences = local_text_embed.size(0)
            fig, axes = plt.subplots(
                num_sentences, 3,
                figsize=(15, 4 * num_sentences),
                dpi=150,
                gridspec_kw={'width_ratios': [1, 1, 1.5]}
            )
            if num_sentences == 1:
                axes = np.array([axes]).reshape(1, -1)

            img = (batch["image"][idx] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).to('cuda')
            orig_img = batch["image"][idx].cpu().numpy().transpose(1, 2, 0)

            for index in range(num_sentences):
                atten_probs = attention_probs[index].clone()
                atten_probs = (atten_probs - atten_probs.min()) / (atten_probs.max() - atten_probs.min() + 1e-8)
                atten_map = rearrange(1 - atten_probs, "(p1 p2) -> p1 p2", p1=19, p2=19)
                atten_map = atten_map.view(1, 1, 19, 19).to('cuda')
                atten_map = F.interpolate(atten_map, size=[224, 224], mode="bilinear", align_corners=True).squeeze()
                atten_map = gaussian_blur_torch(atten_map)
                atten_map = (atten_map * 255).clamp(0, 255).byte()

                alpha = 0.3
                heatmap = apply_colormap_torch(atten_map, colormap='jet')
                blended = (heatmap * alpha + img * (1 - alpha)).clamp(0, 1)
                blended = (blended * 255).byte().cpu().numpy()

                axes[index, 0].imshow(orig_img)
                axes[index, 0].set_title("Original Image", fontsize=12)
                axes[index, 0].axis("off")

                axes[index, 1].imshow(blended)
                axes[index, 1].set_title("Attention Heatmap", fontsize=12)
                axes[index, 1].axis("off")

                axes[index, 2].axis("off")
                sentence = sentences_list[index]
                text_bbox = dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5')
                axes[index, 2].text(
                    0.5, 0.5, sentence,
                    fontsize=12,
                    color='white',
                    verticalalignment='center',
                    horizontalalignment='center',
                    wrap=True,
                    bbox=text_bbox
                )

            plt.tight_layout()
            save_path = os.path.join('/data', f"HAMeR/visualization_8_gpu_180")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'visualization_batch_{batch_idx}.png'), bbox_inches='tight')
            plt.close(fig)











