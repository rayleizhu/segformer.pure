import torch
import torch.nn as nn
from typing import List

from .mit import MiTB0, MiTB1, MiTB2, MiTB3, MiTB4, MiTB5
from .head import SegFormerHead

from torch.hub import load_state_dict_from_url
import os


__all__ = [
    "SegFormer",
    "SegFormerB0",
    "SegFormerB1",
    "SegFormerB2",
    "SegFormerB3",
    "SegFormerB4",
    "SegFormerB5"
]


# https://github.com/pytorch/pytorch/issues/34850
# official checkpoints: https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA
# conversion site: https://sites.google.com/site/gdocs2direct/
# model_urls = {
#     #"segformer.b0.512x512.ade.160k": "https://drive.google.com/uc?export=download&id=1je1GL6TXU3U-cZZsUv08ITUkVW4mBPYy",
# }



def _get_encoder_from_name(name:str, weight=None, in_ch=3)->nn.Module:
    if name == 'mit_b0':
        model = MiTB0()
    elif name == 'mit_b1':
        model = MiTB1()
    elif name == 'mit_b2':
        model = MiTB2()
    elif name == 'mit_b3':
        model = MiTB3()
    elif name == 'mit_b4':
        model = MiTB4()
    elif name == 'mit_b5':
        model = MiTB5()
    else:
        raise ValueError(f'Unsupported encoder name {name}!')
    
    if weight is not None:
        assert weight == 'imagenet', "only imagenet pretrained weight is available currently!"
        model.load_official_state_dict(name+'.pth')
    
    if in_ch != 3:
        model.reset_input_channel(new_in_chans=in_ch, pretrained=(weight is not None))

    return model



class SegFormer(nn.Module):
    official_ckpts = {}
    def __init__(self, in_ch=3,
               encoder_name:str='mit_b0',
               encoder_weight:str=None,
               in_channels=[32, 64, 160, 256],
               in_index=[0, 1, 2, 3],
               feature_strides=[4, 8, 16, 32],
               dropout_ratio=0.1,
               num_classes=19,
               embedding_dim=256):
        super(SegFormer, self).__init__()
        self.backbone = _get_encoder_from_name(encoder_name, weight=encoder_weight, in_ch=in_ch)
        self.decode_head = SegFormerHead(num_classes=num_classes,
                                     in_index=in_index,
                                     in_channels=in_channels,
                                     feature_strides=feature_strides,
                                     embedding_dim=embedding_dim,
                                     dropout_ratio=dropout_ratio
                                     )
    def forward(self, x):
        return self.decode_head(self.backbone(x))

    
    def load_official_state_dict(self, filename:str, local_dir:str=None, download:bool=True, strict:bool=True):
        """
        Args:
            local_dir: if not None, load from "local_dir/filename"
            strict: note that, the definition is silghtly different from load_state_dict().
                    If set to flase, the weight of final layer will not be loaded, so num_classes can be any.
        """

        assert filename in self.official_ckpts.keys(), \
             f"available checkpoints are {self.official_ckpts.keys()}"

        if local_dir is None:
            local_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'checkpoints')
        
        path = os.path.join(local_dir, filename)
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location='cpu')
        elif download:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            import gdown
            url = self.official_ckpts[filename]
            gdown.download(url, path, quiet=False)
            ckpt = torch.load(path, map_location='cpu')
            # ckpt = load_state_dict_from_url(url, progress=True, file_name=filename, model_dir=local_dir)
        else: 
            raise ValueError('You neither proivde local path nor set download True!')
        
        state_dict = ckpt['state_dict']
        # The following two keys are introduced by MMSeg but not used by SegFormer.
        # See https://github.com/NVlabs/SegFormer/blob/9454025f0e74acbbc19c65cbbdf3ff8224997fe3/mmseg/models/decode_heads/decode_head.py#L83
        exclude_keys = ["decode_head.conv_seg.weight", "decode_head.conv_seg.bias"]
        if not strict:
            exclude_keys += ["decode_head.linear_pred.weight", "decode_head.linear_pred.bias"]
        ckpt_to_load = {k:v for k, v in state_dict.items() if k not in exclude_keys}
        self.load_state_dict(ckpt_to_load, strict=strict)




class SegFormerB0(SegFormer):
    official_ckpts = {
        "segformer.b0.512x512.ade.160k.pth": "https://drive.google.com/uc?export=download&id=1je1GL6TXU3U-cZZsUv08ITUkVW4mBPYy",
        "segformer.b0.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=10lD5u0xVDJDKkIYxJDWkSeA2mfK_tgh9"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB0, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b0',
                                          in_channels=[32, 64, 160, 256],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=256)
    
    
class SegFormerB1(SegFormer):
    official_ckpts = {
        "segformer.b1.512x512.ade.160k.pth": "https://drive.google.com/uc?export=download&id=1PNaxIg3gAqtxrqTNsYPriR2c9j68umuj",
        "segformer.b1.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=1sSdiqRsRMhLJCfs0SydF7iKgeQNcXDZj"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB1, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b1',
                                          in_channels=[64, 128, 320, 512],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=256)


class SegFormerB2(SegFormer):
    official_ckpts = {
        "segformer.b2.512x512.ade.160k.pth": "https://drive.google.com/uc?export=download&id=13AMcdZYePbrTtwVzdJwZP5PF8PKehGhU",
        "segformer.b2.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=1MZhqvWDOKdo5rBPC2sL6kWL25JpxOg38"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB2, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b2',
                                          in_channels=[64, 128, 320, 512],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=768)


class SegFormerB3(SegFormer):
    official_ckpts = {
        "segformer.b3.512x512.ade.160k.pth": "https://drive.google.com/uc?export=download&id=16ILNDrZrQRJrXsIcSjUC56ueR72Rlant",
        "segformer.b3.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=1dc1YM2b3844-dLKq0qe77qb9_7brReIF"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB3, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b3',
                                          in_channels=[64, 128, 320, 512],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=768)


# https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py
class SegFormerB4(SegFormer):
    official_ckpts = {
        "segformer.b5.512x512.ade.160k.pth": "https://drive.google.com/uc?export=download&id=171YHhri1rT5lwxmfPW76eU9DPP9OR27n",
        "segformer.b5.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=1F9QqGFzhr5wdX-FWax1xE2l7B8lqs42s"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB4, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b4',
                                          in_channels=[64, 128, 320, 512],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=768)



# https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py
class SegFormerB5(SegFormer):
    official_ckpts = {
        "segformer.b5.640x640.ade.160k.pth": "https://drive.google.com/uc?export=download&id=11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz",
        "segformer.b5.1024x1024.city.160k.pth": "https://drive.google.com/uc?export=download&id=1z3eFf-xVMkcb1Nmcibv6Ut-lTh81RLgO"
    }
    def __init__(self, num_classes:int, in_ch=3, encoder_weight: str = None):
        super(SegFormerB5, self).__init__(in_ch=in_ch,
                                          num_classes=num_classes,
                                          encoder_weight=encoder_weight,
                                          encoder_name='mit_b5',
                                          in_channels=[64, 128, 320, 512],
                                          in_index=[0, 1, 2, 3],
                                          feature_strides=[4, 8, 16, 32],
                                          dropout_ratio=0.1,
                                          embedding_dim=768)



# def _segformer(ckpt:str,
#                progress:bool,
#                encoder_name:str,
#                in_channels:List[int],
#                in_index:List[int],
#                feature_strides:List[int],
#                dropout_ratio:float,
#                num_classes:int,
#                embedding_dim:int):
#     model = SegFormer(encoder_name,
#                       in_channels,
#                       in_index,
#                       feature_strides,
#                       dropout_ratio,
#                       num_classes,
#                       embedding_dim)
#     if ckpt is not None:
#         filename = '_'.join(ckpt.split('.'))+'.pth'
#         state_dict = load_state_dict_from_url(model_urls[ckpt], progress=progress,
#                                               file_name=filename)
#         model.load_state_dict(state_dict)
#     return model


# def remove_unused_from_state_dict(state_dict):
#     """
#     This function is to remove unused layer in model's state dict
#     To matain compatiability to officially released checkpoints
#     """
#     # The following two keys are introduced by MMSeg but not used by SegFormer.
#     # See https://github.com/NVlabs/SegFormer/blob/9454025f0e74acbbc19c65cbbdf3ff8224997fe3/mmseg/models/decode_heads/decode_head.py#L83
#     exclude_keys = ["decode_head.conv_seg.weight", "decode_head.conv_seg.bias"] 
#     return {k:v for k, v in state_dict.items() if k not in exclude_keys }


# def segformer_b0(ckpt:str=None, progress:bool=True):
#     if ckpt is not None:
#         online_available_ckpts = [x for x in model_urls.keys() if x.split('.')[1]=='b0']
#         assert ckpt in online_available_ckpts, f"available checkpoints are {online_available_ckpts}"
#     return  _segformer(encoder_name='mit_b0',
#                        in_channels=[32, 64, 160, 256],
#                        in_index=[0, 1, 2, 3],
#                        feature_strides=[4, 8, 16, 32],
#                        dropout_ratio=0.1,
#                        num_classes=150,
#                        embedding_dim=256,
#                        ckpt=ckpt,
#                        progress=progress
#                      )