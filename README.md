

This is a cleaned version of [official SegFormer](https://github.com/NVlabs/SegFormer). It removes dependency on MMCV and MMSegmentation, which use deep wrapings.

## Requirements

* pytorch>=1.0
* timm>=0.5.4
* gdown (optional, only required if you want to automatically load official checkpoints from url)


## Features

- [x] written with pure pytorch api, no deep wraping, easy to understand, modification friendly
- [x] compatiable with officially released model weights
    - [SegFormer weights](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA)
    - [MixVisionTransformer weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia)
- [x] automatically downloads official checkpoints online

Example usage:

```python
from networks.segformer import *
import torch

model1 = SegFormerB0(num_classes=150, encoder_weight=None)
print(model1.official_ckpts) # print officially released checkpoints
model1.load_official_state_dict('segformer.b0.512x512.ade.160k.pth', strict=True) # load official released weights

model2 = SegFormerB0(num_classes=1, encoder_weight=None) # binary classifier
model2.load_official_state_dict('segformer.b0.512x512.ade.160k.pth', strict=False) # the final prediction layer is not loaded

model3 = SegFormerB0(num_classes=20, encoder_weight='imagenet') # load only ImageNet-pretained backbone

x = torch.zeros((2, 3, 512, 512))
pred = model3(x)
print(pred.size()) # final resolution is (h/4, w/4)
```

## TODOs

- [x] MixVisionTransformer.load_official_state_dict()
- [ ] Flexible input channels for ImageNet pretained MiT()
