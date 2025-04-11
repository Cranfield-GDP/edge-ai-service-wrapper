---
license: apache-2.0
---
# Vision Transformer (ViT) for Facial Expression Recognition Model Card

## Model Overview

- **Model Name:** [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)

- **Task:** Facial Expression/Emotion Recognition

- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

- **Model Architecture:** [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)

- **Finetuned from model:** [vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)

## Model Description

The vit-face-expression model is a Vision Transformer fine-tuned for the task of facial emotion recognition. 

It is trained on the FER2013 dataset, which consists of facial images categorized into seven different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Data Preprocessing

The input images are preprocessed before being fed into the model. The preprocessing steps include:
- **Resizing:** Images are resized to the specified input size.
- **Normalization:** Pixel values are normalized to a specific range.
- **Data Augmentation:** Random transformations such as rotations, flips, and zooms are applied to augment the training dataset.

## Evaluation Metrics

- **Validation set accuracy:** 0.7113
- **Test set accuracy:** 0.7116

## Limitations

- **Data Bias:** The model's performance may be influenced by biases present in the training data.
- **Generalization:** The model's ability to generalize to unseen data is subject to the diversity of the training dataset.