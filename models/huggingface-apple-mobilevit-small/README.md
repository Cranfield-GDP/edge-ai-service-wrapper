---
license: other
tags:
- vision
- image-classification
datasets:
- imagenet-1k
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# MobileViT (small-sized model)

MobileViT model pre-trained on ImageNet-1k at resolution 256x256. It was introduced in [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) by Sachin Mehta and Mohammad Rastegari, and first released in [this repository](https://github.com/apple/ml-cvnets). The license used is [Apple sample code license](https://github.com/apple/ml-cvnets/blob/main/LICENSE).

Disclaimer: The team releasing MobileViT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

MobileViT is a light-weight, low latency convolutional neural network that combines MobileNetV2-style layers with a new block that replaces local processing in convolutions with global processing using transformers. As with ViT (Vision Transformer), the image data is converted into flattened patches before it is processed by the transformer layers. Afterwards, the patches are "unflattened" back into feature maps. This allows the MobileViT-block to be placed anywhere inside a CNN. MobileViT does not require any positional embeddings.

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=mobilevit) to look for fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

Currently, both the feature extractor and model support PyTorch.

## Training data

The MobileViT model was pretrained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k), a dataset consisting of 1 million images and 1,000 classes. 

## Training procedure

### Preprocessing

Training requires only basic data augmentation, i.e. random resized cropping and horizontal flipping. 

To learn multi-scale representations without requiring fine-tuning, a multi-scale sampler was used during training, with image sizes randomly sampled from: (160, 160), (192, 192), (256, 256), (288, 288), (320, 320).

At inference time, images are resized/rescaled to the same resolution (288x288), and center-cropped at 256x256.

Pixels are normalized to the range [0, 1]. Images are expected to be in BGR pixel order, not RGB.

### Pretraining

The MobileViT networks are trained from scratch for 300 epochs on ImageNet-1k on 8 NVIDIA GPUs with an effective batch size of 1024 and learning rate warmup for 3k steps, followed by cosine annealing. Also used were label smoothing cross-entropy loss and L2 weight decay. Training resolution varies from 160x160 to 320x320, using multi-scale sampling.

## Evaluation results

| Model            | ImageNet top-1 accuracy | ImageNet top-5 accuracy | # params  | URL                                             |
|------------------|-------------------------|-------------------------|-----------|-------------------------------------------------|
| MobileViT-XXS    | 69.0                    | 88.9                    | 1.3 M     | https://huggingface.co/apple/mobilevit-xx-small |
| MobileViT-XS     | 74.8                    | 92.3                    | 2.3 M     | https://huggingface.co/apple/mobilevit-x-small  |
| **MobileViT-S**  | **78.4**                | **94.1**                | **5.6 M** | https://huggingface.co/apple/mobilevit-small    |

### BibTeX entry and citation info

```bibtex
@inproceedings{vision-transformer,
title = {MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
author = {Sachin Mehta and Mohammad Rastegari},
year = {2022},
URL = {https://arxiv.org/abs/2110.02178}
}
```
