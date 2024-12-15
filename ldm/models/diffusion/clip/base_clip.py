import torch
import torch.nn as nn
from .clip import clip
# from clip import clip
import torchvision
from PIL import Image

# model_name = "ViT-B/16"
model_name = "ViT-B/32"

def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda:1").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda:1")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        device="cuda:1"
        self.clip_model = load_clip_to_cpu()
        # self.clip_model = imagebind_model.imagebind_huge(pretrained=True).to(device)
        self.clip_model.requires_grad_(False)
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.device = device
        self = self.to(device)
    
    @torch.no_grad()
    def cal_ref(self, ref_path):

        img = Image.open(ref_path).convert('RGB')
        self.raw_ref = img

        img = self.to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        self.ref = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear')
        self.f2, feat2 = self.clip_model.encode_image_with_features(self.ref, pick=2)
        feat2 = feat2[1:, 0, :]
        self.gram2 = torch.mm(feat2.t(), feat2)

        text = ref_path.split("/")[-2]
        text = " ".join(text.split(" ")[1:])
        text = clip.tokenize([text]).to(self.device)
        self.text_emb = self.clip_model.encode_text(text)


    def enc_func(self, im, bbox=None):
        im0 = self.preprocess(im).to(self.device)
        im1 = torch.nn.functional.interpolate(im0, size=(224, 224), mode='bicubic')
        f1, feat1 = self.clip_model.encode_image_with_features(im1, pick=2)

        loss = 1- torch.cosine_similarity(f1, self.text_emb.detach(), dim=1).mean()

        feat1 = feat1[1:, 0, :]
        gram1 = torch.mm(feat1.t(), feat1)
        gram = torch.linalg.norm(gram1-self.gram2.detach())
        return loss, gram, 0  