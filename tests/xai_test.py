from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from transformers import AutoImageProcessor
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
from transformers import ResNetForImageClassification
from PIL import Image

MODEL_NAME = "microsoft/resnet-50"
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
resize_and_normalize_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, use_fast=True
)
model.eval()
id2label = model.config.id2label


resize_only_processor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ],
)

image_path = "puppy.png"
image = Image.open(image_path).convert("RGB")

normalized_image_tensor = resize_and_normalize_processor(
    images=image, return_tensors="pt"
)["pixel_values"].squeeze(0)
original_image_tensor = resize_only_processor(image)


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.
"""


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""


def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    targets_for_gradcam: Optional[List[Callable]],
    reshape_transform: Optional[Callable],
    input_tensor: torch.nn.Module = None,
    input_image: Image = None,
    method: Callable = GradCAM,
):
    
    with method(
        model=HuggingfaceToTensorModelWrapper(model),
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        print("shape of input tensor", input_tensor.shape)

        # # the shape of input tensor is (3, 500, 599)
        # # now save the input tensor as an image into a file
        # cv2.imwrite("input_tensor.png", input_tensor.permute(1, 2, 0).numpy() * 255)

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(
            (
                1
                if targets_for_gradcam is None or len(targets_for_gradcam) == 0
                else len(targets_for_gradcam)
            ),
            1,
            1,
            1,
        )

        batch_results = cam(
            input_tensor=repeated_tensor,
            targets=(
                None
                if targets_for_gradcam is None or len(targets_for_gradcam) == 0
                else targets_for_gradcam
            ),
        )
        results = []
        for grayscale_cam in batch_results:
            # adjust the shape of the input_image from (3, 244, 244) to (244, 244, 3)
            visualization = show_cam_on_image(
                np.float32(input_image.permute(1, 2, 0).numpy()), grayscale_cam, use_rgb=True
            )
            results.append(visualization)
        return np.hstack(results), cam.outputs


def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")


# We will show GradCAM for the "Egyptian Cat" and the 'Remote Control" categories:
targets_for_gradcam = [
    ClassifierOutputTarget(151),
    ClassifierOutputTarget(968),
]

# The last layer in the Resnet Encoder:
target_layer = model.resnet.encoder.stages[-1].layers[-1]


vis_output, model_outputs = run_grad_cam_on_image(
        model=model,
        input_tensor=normalized_image_tensor,
        target_layer=target_layer,
        input_image=original_image_tensor,
        targets_for_gradcam=targets_for_gradcam,
        reshape_transform=None,
    )
output_img = Image.fromarray(vis_output)

# save the image
output_img.save("xai_output.png")

probabilities = torch.nn.functional.softmax(model_outputs[0], dim=0)

# Return the top 5 predictions with labels
top5_prob, top5_catid = torch.topk(probabilities, 5)
predictions = []
for i in range(top5_prob.size(0)):
    category_id = top5_catid[i].item()
    predictions.append(
        {
            "category_id": category_id,
            "label": id2label[category_id],
            "probability": top5_prob[i].item(),
        }
    )
print("Predictions:")
print(predictions)

# print_top_categories(model, original_image_tensor)
