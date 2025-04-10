from typing import Callable, List, Optional
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torch.profiler import profile, record_function
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    AblationCAM,
    XGradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    LayerCAM,
    EigenCAM,
    EigenGradCAM,
    KPCA_CAM,
    RandomCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor
from typing import List, Optional


# import model utilities
from ai_server_utils import (
    encode_image,
    prepare_profile_results,
    process_model_output_logits,
    profile_activities,
)

# Currently only support GradCAM on image-classification models.
# so we import the model directly from the model.py file
from model import model, MODEL_NAME, device


resize_and_normalize_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, use_fast=True
)
resize_only_processor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ],
)

GRADCAM_METHODS = {
    "GradCAM": GradCAM,
    "HiResCAM": HiResCAM,
    "AblationCAM": AblationCAM,
    "XGradCAM": XGradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "ScoreCAM": ScoreCAM,
    "LayerCAM": LayerCAM,
    "EigenCAM": EigenCAM,
    "EigenGradCAM": EigenGradCAM,
    "KPCA_CAM": KPCA_CAM,
    "RandomCAM": RandomCAM,
}


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    """Model wrapper to return a tensor"""

    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def get_model_to_tensor_wrapper_class():
    """Helper function to get the model wrapper class."""
    return HuggingfaceToTensorModelWrapper


def get_target_layers_for_grad_cam(model: torch.nn.Module):
    """Helper function to get the target layer for GradCAM."""
    return [model.resnet.encoder.stages[-1].layers[-1]]


def get_classifier_output_target_class():
    """Helper function to get the classifier output target class."""
    return ClassifierOutputTarget


def get_reshape_transform():
    """Helper function to get the reshape transform for GradCAM."""
    return None


def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layers: List[torch.nn.Module],
    targets_for_gradcam: Optional[List[Callable]],
    reshape_transform: Optional[Callable],
    input_tensor: torch.nn.Module,
    input_image: Image,
    gradcam_method: Callable,
):
    """Helper function to run GradCAM on an image and create a visualization.
    Since the classification target is None, the highest scoring category will be used for every image in the batch.
    """

    with gradcam_method(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:

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
                np.float32(input_image.permute(1, 2, 0).numpy()),
                grayscale_cam,
                use_rgb=True,
            )
            results.append(visualization)
        output_image = Image.fromarray(np.hstack(results))

        return output_image, cam.outputs


# Initialize the FastAPI router
router = APIRouter()


@router.post("/run")
async def run_model(
    file: UploadFile = File(...),
    ue_id: str = Form(...),
    gradcam_method_name: str = Form(...),
    target_category_indexes: Optional[List[int]] = Form(None),
):
    """
    Endpoint to run the XAI model."""

    try:
        # Prepare the model input
        print("Preparing the model input...")
        image = Image.open(file.file).convert("RGB")
        normalized_image_tensor = resize_and_normalize_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0).to(device)
        original_image_tensor = resize_only_processor(image).to(device)

        if target_category_indexes is None or len(target_category_indexes) == 0:
            targets_for_gradcam = None
        else:
            # Convert to output target from category indexes
            targets_for_gradcam = [
                ClassifierOutputTarget(index) for index in target_category_indexes
            ]
        assert (
            gradcam_method_name in GRADCAM_METHODS
        ), f"GradCAM method '{gradcam_method_name}' is not supported. "
        gradcam_method = GRADCAM_METHODS[gradcam_method_name]

        model_wrapper_class = get_model_to_tensor_wrapper_class()
        target_layers = get_target_layers_for_grad_cam(model)
        reshape_transform = get_reshape_transform()

        # Perform inference
        print("Running GradCAM...")
        xai_image, model_output_logits = run_grad_cam_on_image(
            model=model_wrapper_class(model),
            target_layers=target_layers,
            targets_for_gradcam=targets_for_gradcam,
            reshape_transform=reshape_transform,
            input_tensor=normalized_image_tensor,
            input_image=original_image_tensor,
            gradcam_method=gradcam_method,
        )

        predictions = process_model_output_logits(model, model_output_logits)

        return JSONResponse(
            {
                "ue_id": ue_id,
                "xai_results": {
                    "image": encode_image(xai_image),
                    "xai_method": gradcam_method_name,
                },
                "model_results": predictions,
            }
        )

    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )


@router.post("/profile_run")
async def profile_run(
    file: UploadFile = File(...),
    ue_id: str = Form(...),
    gradcam_method_name: str = Form(...),
    target_category_indexes: Optional[List[int]] = Form(None),
):
    """
    Endpoint to profile the XAI run.
    """
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        normalized_image_tensor = resize_and_normalize_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0).to(device)
        original_image_tensor = resize_only_processor(image).to(device)
        if target_category_indexes is None or len(target_category_indexes) == 0:
            targets_for_gradcam = None
        else:
            # Convert to output target from category indexes
            targets_for_gradcam = [
                ClassifierOutputTarget(index) for index in target_category_indexes
            ]

        assert (
            gradcam_method_name in GRADCAM_METHODS
        ), f"GradCAM method '{gradcam_method_name}' is not supported. "
        gradcam_method = GRADCAM_METHODS[gradcam_method_name]

        model_wrapper_class = get_model_to_tensor_wrapper_class()
        target_layers = get_target_layers_for_grad_cam(model)
        reshape_transform = get_reshape_transform()

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("xai_model_run"):

                # Perform inference
                xai_image, model_output_logits = run_grad_cam_on_image(
                    model=model_wrapper_class(model),
                    target_layers=target_layers,
                    targets_for_gradcam=targets_for_gradcam,
                    reshape_transform=reshape_transform,
                    input_tensor=normalized_image_tensor,
                    input_image=original_image_tensor,
                    gradcam_method=gradcam_method,
                )

        return JSONResponse(
            {
                "ue_id": ue_id,
                "xai_results": {
                    "image": encode_image(xai_image),
                    "xai_method": gradcam_method_name,
                },
                "model_results": process_model_output_logits(
                    model, model_output_logits
                ),
                "profile_result": prepare_profile_results(prof),
            }
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            content={"error": f"Failed to process the request. {e}"},
            status_code=500,
        )


XAI_OUTPUT_JSON_SPEC = {
    "xai_results": {
        "image": "XAI image result",
        "xai_method": "XAI method used",
    }
}
