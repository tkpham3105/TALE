import torch
from torchvision import transforms

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_normalization(content_feat_ori, style_feat, roi=None, n_patch=3, alpha=1, mask=None):
    assert (content_feat_ori.size()[:2] == style_feat.size()[:2])
    
    if roi is not None:
        content_feat = content_feat_ori[:, :, roi[0]:roi[1], roi[2]:roi[3]].clone()
        size = content_feat.size()
        content_feat = content_feat[mask]
    else:
        content_feat = content_feat_ori.clone()
        size = content_feat.size()
    scale = 1 / n_patch
    crop_size = (size[0], size[1], int(size[2] * scale), int(size[3] * scale))
    cropper = transforms.RandomCrop(crop_size[2:])

    feat_var = content_feat.view(1, 4, -1).var(dim=2) + 1e-5
    feat_std = feat_var.sqrt().view(1, 4, 1)
    feat_mean = content_feat.view(1, 4, -1).mean(dim=2).view(1, 4, 1)
    normalized_feat = (content_feat.view(1, 4, -1) - feat_mean) / feat_std
    style_mean, style_std = calc_mean_std(style_feat.clone())    
    content_feat = normalized_feat * style_std.squeeze(-1) + style_mean.squeeze(-1)
    content_out = content_feat_ori.clone()
    
    if roi is not None:
        content_out[:, :, roi[0]:roi[1], roi[2]:roi[3]][mask] = alpha * content_feat.flatten() + (1-alpha) * content_feat_ori[:, :, roi[0]:roi[1], roi[2]:roi[3]][mask]
    else:
        content_out = alpha * content_feat + (1-alpha) * content_feat_ori

    return content_out
