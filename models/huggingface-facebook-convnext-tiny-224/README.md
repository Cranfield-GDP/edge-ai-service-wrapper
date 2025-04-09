---
license: apache-2.0
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

# ConvNeXT (tiny-sized model) 

ConvNeXT model trained on ImageNet-1k at resolution 224x224. It was introduced in the paper [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) by Liu et al. and first released in [this repository](https://github.com/facebookresearch/ConvNeXt). 

Disclaimer: The team releasing ConvNeXT did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

ConvNeXT is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform them. The authors started from a ResNet and "modernized" its design by taking the Swin Transformer as inspiration.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=convnext) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/convnext).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2201-03545,
  author    = {Zhuang Liu and
               Hanzi Mao and
               Chao{-}Yuan Wu and
               Christoph Feichtenhofer and
               Trevor Darrell and
               Saining Xie},
  title     = {A ConvNet for the 2020s},
  journal   = {CoRR},
  volume    = {abs/2201.03545},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.03545},
  eprinttype = {arXiv},
  eprint    = {2201.03545},
  timestamp = {Thu, 20 Jan 2022 14:21:35 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-03545.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```