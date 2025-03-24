#!/usr/bin/env python
"""
VideoT1 Multi-GPU Inference.
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output (use with caution in production)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyramid_flow.pyramid_dit import PyramidDiTForVideoGeneration
from pipeline.videot1_pipeline import VideoT1Generator


# --- Model and Parameter Configuration ---
# Please change them into **your** configurations
# Base Model
MODEL_DTYPE = 'bf16'  # Data type for model weights
MODEL_NAME = "pyramid_flux"  # Model selection: "pyramid_flux" or "pyramid_mmdit"
DEFAULT_MODEL_PATH = "/mnt/public/huggingface/pyramid-flow-miniflux/"  # Default path to Pyramid-Flow checkpoint
# VisionReward Model
DEFAULT_VR_PATH = '/mnt/public/huggingface/VisionReward-Video'  # Default path to VisionReward model
# Image-CoT Model
DEFAULT_IMGCOT_PATH = "/mnt/public/huggingface/Image-Generation-CoT/parm"
# Result/Intermediate Files Path
DEFAULT_RESULT_PATH = "./final_results"  # Default directory to save final videos
DEFAULT_INTERMED_PATH = "./intermed_files" # Default directory to save intermediate files
# LM for Hierarchical Prompts
DEFAULT_LM_PATH = "/mnt/public/huggingface/DeepSeek-R1-Distill-Llama-8B"


def parse_args():
    parser = argparse.ArgumentParser(description="Video Generation with VideoCoT")

    # --- Essential Arguments ---
    parser.add_argument(
        "--prompt",
        default="A cat wearing sunglasses and working as a lifeguard at a pool.",
        type=str,
        help="Text prompt for video generation. "
             "This prompt guides the content of the generated video."
    )
    parser.add_argument(
        "--video_name",
        default=None,
        type=str,
        help="Name of the video file (without extension) for saving the generated video."
    )

    # --- Branch Configuration ---
    parser.add_argument(
        "--img_branch",
        type=str,
        default="5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        help="Comma-separated number of image branches at each level of the model. "
             "Example: '5,1,2'. Defines the image generation structure."
    )
    parser.add_argument(
        "--video_branch",
        type=str,
        default="5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        help="Comma-separated number of video branches at each level of the model. "
             "Example: '3,2,2'. Defines the video generation structure."
    )

    # --- Optional Parameters ---
    parser.add_argument(
        "--temp",
        type=int,
        default=16,
        help="Number of frames to generate in the video. "
             f"Maximum is 16 for the 384p variant. Default is {16}."
    )
    parser.add_argument(
        "--base_device_id",
        type=int,
        default=1,
        help="CUDA device ID to use for generation. Default is 1."
    )
    parser.add_argument(
        "--imgcot_device_id",
        type=int,
        default=2,
        help="CUDA device ID to use for generation. Default is 2."
    )
    parser.add_argument(
        "--reward_device_id",
        type=int,
        default=0,
        help="CUDA device ID to use for generation. Default is 0."
    )
    parser.add_argument(
        "--lm_device_id",
        type=int,
        default=3,
        help="CUDA device ID to use for generation. Default is 3."
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default=DEFAULT_RESULT_PATH,
        help=f"Directory to save the final generated videos. Default is '{DEFAULT_RESULT_PATH}'."
    )
    parser.add_argument(
        "--intermed_path",
        type=str,
        default=DEFAULT_INTERMED_PATH,
        help=f"Directory to save intermediate images and videos during generation. Default is '{DEFAULT_INTERMED_PATH}'."
    )
    parser.add_argument(
        "--reward_stages",
        type=str,
        default="4,9,13",
        help="Comma-separated list of stages at which to apply the reward model. "
             "Example: '4,9,13'. These stages influence video quality."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="384",
        choices=["384", "768"],
        help="Resolution variant for generated videos. "
             "'384' for lower resolution (640x384), '768' for higher resolution (1280x768). Default is '384'."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the Pyramid-Flow model checkpoint directory. Default is '{DEFAULT_MODEL_PATH}'."
    )
    parser.add_argument(
        "--vr_path",
        type=str,
        default=DEFAULT_VR_PATH,
        help=f"Path to the VisionReward-Video model checkpoint directory. Default is '{DEFAULT_VR_PATH}'."
    )
    parser.add_argument(
        "--imgcot_path",
        type=str,
        default=DEFAULT_IMGCOT_PATH,
        help=f"Path to the VisionReward-Video model checkpoint directory. Default is '{DEFAULT_IMGCOT_PATH}'."
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        default=DEFAULT_LM_PATH,
        help=f"Path to LLM used for hierarchical prompts generation. Default is '{DEFAULT_LM_PATH}'"
    )
    return parser.parse_args()


def validate_params(args):
    if args.temp > 16 and args.variant == "384":
        raise ValueError("For 384p variant, temp (number of frames) must be <= 16.")

    try:
        img_branch = [int(x) for x in args.img_branch.split(',')]
        if not img_branch:  # Check if the list is empty
            raise ValueError
    except ValueError:
        raise ValueError("img_branch must be comma-separated integers, e.g., '5,1,2'.")

    try:
        video_branch = [int(x) for x in args.video_branch.split(',')]
        if not video_branch: # Check if the list is empty
            raise ValueError
    except ValueError:
        raise ValueError("video_branch must be comma-separated integers, e.g., '3,2,2'.")

    try:
        reward_stages = [int(x) for x in args.reward_stages.split(',')]
    except ValueError:
        raise ValueError("reward_stages must be comma-separated integers, e.g., '4,9,13'.")

    os.makedirs(args.result_path, exist_ok=True) # Use result_path here as it is where final videos will be saved.
    os.makedirs(args.intermed_path, exist_ok=True) # Ensure intermediate path exists as well.

    return img_branch, video_branch, reward_stages


def init_pyramid_model(model_path, device, model_variant="diffusion_transformer_384p", model_name="pyramid_flux"):
    
    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype=MODEL_DTYPE,
        model_name=model_name,
        model_variant=model_variant,
        frame_selection_callback=True,  
        video_sync_group=8,            
    )

    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)
    model.vae.enable_tiling() # Enable VAE tiling for memory efficiency, especially for larger images.
    return model


def init_vr_model(model_path, device):
    # Determine torch dtype based on device CUDA capability for VisionReward model
    device_index = int(device.split(':')[-1]) # Extract device index if CUDA device
    torch.cuda.set_device(device_index)
    vr_dtype_local = (torch.bfloat16 if torch.cuda.is_available() and
                                        torch.cuda.get_device_capability(device_index) >= (8,)
                                        else torch.float16) # Use bfloat16 if device supports it (better precision, potentially faster), else float16.

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True # Trust remote code for tokenizer (ensure safety when using external models)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=vr_dtype_local,
        trust_remote_code=True # Trust remote code for model (ensure safety when using external models)
    ).eval().to(device) # Load model, set to evaluation mode, and move to device

    return model, tokenizer



def main():

    args = parse_args()
    print("Running video generation with arguments:", args) # Log arguments at the start for reproducibility

    # --- Resolution and Model Variant ---
    if args.variant == "384":
        width = 640
        height = 384
        model_variant = "diffusion_transformer_384p"
    elif args.variant == "768":
        width = 1280
        height = 768
        model_variant = "diffusion_transformer_768p"
    else: # Should not reach here due to argparse choices, but for robustness:
        raise ValueError(f"Invalid variant: {args.variant}. Must be '384' or '768'.")

    temp = args.temp
    img_branch, video_branch, reward_stages = validate_params(args)
    prompt = args.prompt
    result_path = args.result_path
    video_name = args.video_name
    if not video_name:
        video_name = prompt.replace(' ', '_')
    intermed_path = args.intermed_path

    # --- PF Setup ---
    torch.cuda.set_device(args.base_device_id) # Set CUDA device to be used
    base_device = f"cuda:{args.base_device_id}" if torch.cuda.is_available() else "cpu" # Determine device string
    model_path = args.model_path
    print("Initializing Pyramid model...")
    pyramid_model = init_pyramid_model(model_path, base_device, model_variant)

    # --- VR Setup --- 
    vr_path = args.vr_path       # Get VR model path from arguments
    torch.cuda.set_device(args.reward_device_id)
    reward_device = f"cuda:{args.reward_device_id}" if torch.cuda.is_available() else "cpu" # Determine device string
    print("Initializing VisionReward model...")
    reward_model, tokenizer = init_vr_model(vr_path, reward_device)

    reward_params = {
        "model": reward_model,
        "tokenizer": tokenizer,
    }

    # --- LM Config ---
    lm_path = args.lm_path,
    lm_device = f"cuda:{args.lm_device_id}" if torch.cuda.is_available() else "cpu" # Determine device string

    # --- Video T1 Generation ---
    torch.cuda.set_device(args.base_device_id)
    # Generator Setup
    imgcot_device = f"cuda:{args.imgcot_device_id}" if torch.cuda.is_available() else "cpu" # Determine device string
    generator = VideoT1Generator(
        pyramid_model,
        base_device,
        dtype=torch.bfloat16, # Use bfloat16 for generator computations
        image_selector_path=args.imgcot_path,
        result_path=args.result_path,
        imgcot_device=imgcot_device,
        vr_device=reward_device,
        lm_path=lm_path,
        lm_device=lm_device,
    )


    # --- Video Generation ---

    print("Starting video generation process...")
    best_video = generator.videot1_gen(
        prompt=prompt,
        num_inference_steps=[20, 20, 20],      # Inference steps for image branch at each level
        video_num_inference_steps=[20, 20, 20], # Inference steps for video branch at each level
        height=height,
        width=width,
        num_frames=temp,
        guidance_scale=7.0,           # Guidance scale for image branch
        video_guidance_scale=5.0,      # Guidance scale for video branch
        save_memory=True,             # Enable memory saving optimizations
        inference_multigpu=True,      # Enable multi-GPU inference if available
        video_branching_factors=video_branch,
        image_branching_factors=img_branch,
        reward_stages=reward_stages,
        use_hierarchical_prompts=True,     # Use hierarchical prompts (purpose needs clarification if open-sourced)
        result_path=result_path,
        intermediate_path=intermed_path,
        video_name=video_name,
        **reward_params                # Pass reward model and tokenizer to the generator
    )

    print(f"Video generation completed for prompt: '{prompt}'. Best video saved at: {best_video}")
    print(f"Results are saved in: {result_path}") # Inform user about result path

    return 0


if __name__ == "__main__":
    exit(main())
