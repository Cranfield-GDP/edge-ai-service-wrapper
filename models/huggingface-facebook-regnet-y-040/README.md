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

# RegNet

RegNet model trained on imagenet-1k. It was introduced in the paper [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) and first released in [this repository](https://github.com/facebookresearch/pycls). 

Disclaimer: The team releasing RegNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The authors design search spaces to perform Neural Architecture Search (NAS). They first start from a high dimensional search space and iteratively reduce the search space by empirically applying constraints based on the best-performing models sampled by the current search space.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/regnet_architecture.png)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=regnet) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model:

```python
>>> from transformers import AutoFeatureExtractor, RegNetForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-040")
>>> model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

>>> inputs = feature_extractor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
'tabby, tabby cat'
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/regnet).