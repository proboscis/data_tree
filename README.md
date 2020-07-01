# data_tree
Data science with Series and Tables processing are made back-traceable.

# Tutorial
## Series Example
### scan_image
![example](https://i.gyazo.com/8611701a1213b844bd62b978bac58b81.png)

## Auto Data Conversion Example
```python
from data_tree import auto
from PIL import Image
img_path = "anything.png"
auto_img = auto("image,RGB,RGB")(Image.open(img_path))
tensor_img = auto_img.to("torch,float32,BCHW,RGB,0_255")
```
Please use pytest and see examples in test/ for now.

# TODO
- documentation
- index bug fix
- write simple tutorial

# Configuration
- place `~/.data_tree/config.yml`
  - place visdom.username & visdom.server_address