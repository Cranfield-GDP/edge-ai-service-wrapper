---
license: apache-2.0
pipeline_tag: image-classification
---
# Model Card: Fine-Tuned Vision Transformer (ViT) for NSFW Image Classification

## Model Description

The **Fine-Tuned Vision Transformer (ViT)** is a variant of the transformer encoder architecture, similar to BERT, that has been adapted for image classification tasks. This specific model, named "google/vit-base-patch16-224-in21k," is pre-trained on a substantial collection of images in a supervised manner, leveraging the ImageNet-21k dataset. The images in the pre-training dataset are resized to a resolution of 224x224 pixels, making it suitable for a wide range of image recognition tasks.

During the training phase, meticulous attention was given to hyperparameter settings to ensure optimal model performance. The model was fine-tuned with a judiciously chosen batch size of 16. This choice not only balanced computational efficiency but also allowed for the model to effectively process and learn from a diverse array of images.

To facilitate this fine-tuning process, a learning rate of 5e-5 was employed. The learning rate serves as a critical tuning parameter that dictates the magnitude of adjustments made to the model's parameters during training. In this case, a learning rate of 5e-5 was selected to strike a harmonious balance between rapid convergence and steady optimization, resulting in a model that not only learns swiftly but also steadily refines its capabilities throughout the training process.

This training phase was executed using a proprietary dataset containing an extensive collection of 80,000 images, each characterized by a substantial degree of variability. The dataset was thoughtfully curated to include two distinct classes, namely "normal" and "nsfw." This diversity allowed the model to grasp nuanced visual patterns, equipping it with the competence to accurately differentiate between safe and explicit content.

The overarching objective of this meticulous training process was to impart the model with a deep understanding of visual cues, ensuring its robustness and competence in tackling the specific task of NSFW image classification. The result is a model that stands ready to contribute significantly to content safety and moderation, all while maintaining the highest standards of accuracy and reliability.
## Intended Uses & Limitations

### Intended Uses
- **NSFW Image Classification**: The primary intended use of this model is for the classification of NSFW (Not Safe for Work) images. It has been fine-tuned for this purpose, making it suitable for filtering explicit or inappropriate content in various applications.

### How to use
Here is how to use this model to classifiy an image based on 1 of 2 classes (normal,nsfw):

```markdown

# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline

img = Image.open("<path_to_image_file>")
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
classifier(img)

```

<hr>

``` markdown

# Load model directly
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

img = Image.open("<path_to_image_file>")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]

```

<hr>

### Limitations
- **Specialized Task Fine-Tuning**: While the model is adept at NSFW image classification, its performance may vary when applied to other tasks.
- Users interested in employing this model for different tasks should explore fine-tuned versions available in the model hub for optimal results.

## Training Data

The model's training data includes a proprietary dataset comprising approximately 80,000 images. This dataset encompasses a significant amount of variability and consists of two distinct classes: "normal" and "nsfw." The training process on this data aimed to equip the model with the ability to distinguish between safe and explicit content effectively.

### Training Stats
``` markdown

- 'eval_loss': 0.07463177293539047,
- 'eval_accuracy': 0.980375, 
- 'eval_runtime': 304.9846, 
- 'eval_samples_per_second': 52.462, 
- 'eval_steps_per_second': 3.279

```

<hr>


**Note:** It's essential to use this model responsibly and ethically, adhering to content guidelines and applicable regulations when implementing it in real-world applications, particularly those involving potentially sensitive content.

For more details on model fine-tuning and usage, please refer to the model's documentation and the model hub.

## References

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [ImageNet-21k Dataset](http://www.image-net.org/)

**Disclaimer:** The model's performance may be influenced by the quality and representativeness of the data it was fine-tuned on. Users are encouraged to assess the model's suitability for their specific applications and datasets.