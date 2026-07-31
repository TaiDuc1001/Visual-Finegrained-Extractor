import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class SSLHead(nn.Module):
    def __init__(self, feature_dim, proj_dim=256, num_prototypes=4096):
        super().__init__()
        self.proj = nn.Linear(feature_dim, proj_dim)
        self.proto = nn.Linear(proj_dim, num_prototypes, bias=False)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        u = self.proto(z)
        return u


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FusionWeightLearner(nn.Module):
    def __init__(self, num_classes=None, init_w1=1.0, init_w2=1.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(init_w1))
        self.w2 = nn.Parameter(torch.tensor(init_w2))

    def forward(self, logits_apt, logits_img):
        device = self.w1.device
        logits_apt = logits_apt.to(device)
        logits_img = logits_img.to(device)
        apt_centered = logits_apt - logits_apt.mean(dim=-1, keepdim=True)
        img_centered = logits_img - logits_img.mean(dim=-1, keepdim=True)
        fused = self.w1 * apt_centered + self.w2 * img_centered
        return fused

    def get_weights(self):
        return self.w1.detach().item(), self.w2.detach().item()


class TransformerAdapter(nn.Module):
    def __init__(self, feature_dim, num_layers=1, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(feature_dim),
                    'qkv': nn.Linear(feature_dim, feature_dim * 3),
                    'proj': nn.Linear(feature_dim, feature_dim),
                    'norm2': nn.LayerNorm(feature_dim),
                    'feed_forward': nn.Linear(feature_dim, feature_dim)
                })
            )
        self.norm = nn.LayerNorm(feature_dim)
        self.last_attn_weights = None

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq_len = x.shape[1]
        attn_weights_all = []
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            qkv = layer['qkv'](x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_weights_all.append(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, seq_len, -1)
            x = layer['proj'](x)
            x = residual + x
            x = layer['feed_forward'](layer['norm2'](residual))
        x = self.norm(x)
        if seq_len == 1:
            x = x.squeeze(1)
        if attn_weights_all:
            self.last_attn_weights = attn_weights_all[-1].detach()
        if return_attention:
            return x, attn_weights_all
        return x


class ImageSSLModel(nn.Module):
    def __init__(self, image_encoder, feature_dim, proj_dim=256, num_prototypes=4096,
                 num_trans_layers=1, num_heads=8):
        super().__init__()
        self.encoder = image_encoder
        self.num_heads = num_heads
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.adapter = TransformerAdapter(feature_dim, num_trans_layers, num_heads)
        self.ssl_head = SSLHead(feature_dim, proj_dim, num_prototypes)

    def forward(self, x, return_attention=False):
        with torch.no_grad():
            visual_output = self.encoder(x)
            if isinstance(visual_output, tuple):
                all_tokens, _ = visual_output
            else:
                all_tokens = visual_output
        encoder_cls = all_tokens[:, 0, :] if all_tokens.dim() == 3 else all_tokens
        if return_attention:
            adapted_tokens, attn_weights = self.adapter(all_tokens, return_attention=True)
            adapted_cls = adapted_tokens[:, 0, :] if adapted_tokens.dim() == 3 else adapted_tokens
            cls_feat = adapted_cls + encoder_cls
            u = self.ssl_head(cls_feat)
            return u, cls_feat, attn_weights
        adapted_tokens = self.adapter(all_tokens)
        adapted_cls = adapted_tokens[:, 0, :] if adapted_tokens.dim() == 3 else adapted_tokens
        cls_feat = adapted_cls + encoder_cls
        u = self.ssl_head(cls_feat)
        return u, cls_feat

    def get_attention_weights(self, x):
        with torch.no_grad():
            visual_output = self.encoder(x)
            if isinstance(visual_output, tuple):
                all_tokens, _ = visual_output
            else:
                all_tokens = visual_output
            _, attn_weights = self.adapter(all_tokens, return_attention=True)
        return attn_weights


class DINOMultiCropTransform:
    def __init__(self, clip_mean, clip_std, global_crop_size=224, local_crop_size=96,
                 global_crop_scale=(0.4, 1.0), local_crop_scale=(0.05, 0.4), num_local_crops=6):
        self.num_local_crops = num_local_crops
        
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
        
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

    def __call__(self, img):
        global_views = [self.global_transform(img), self.global_transform(img)]
        local_views = [self.local_transform(img) for _ in range(self.num_local_crops)]
        return global_views, local_views
