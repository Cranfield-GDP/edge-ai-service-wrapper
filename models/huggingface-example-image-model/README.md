---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
---

# ResNet-50 v1.5

ResNet model pre-trained on ImageNet-1k at resolution 224x224. It was introduced in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by He et al. 

Disclaimer: The team releasing ResNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

ResNet (Residual Network) is a convolutional neural network that democratized the concepts of residual learning and skip connections. This enables to train much deeper models.

This is ResNet v1.5, which differs from the original model: in the bottleneck blocks which require downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution. This difference makes ResNet50 v1.5 slightly more accurate (\~0.5% top1) than v1, but comes with a small performance drawback (~5% imgs/sec) according to [Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch).

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/resnet_architecture.png)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=resnet) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/resnet).

### BibTeX entry and citation info

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
