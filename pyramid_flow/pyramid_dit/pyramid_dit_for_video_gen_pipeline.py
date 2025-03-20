import torch
import os
import gc
import sys
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import math
import random
import PIL
import cv2
import io
import datetime
import json

from queue import SimpleQueue
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
from accelerate import Accelerator, cpu_offload
from ..diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler
from ..video_vae.modeling_causal_vae import CausalVideoVAE
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers.utils import load_image, export_to_video, export_to_gif
from ..trainer_misc import (
    all_to_all,
    is_sequence_parallel_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_group_rank,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_rank,
)

from .mmdit_modules import (
    PyramidDiffusionMMDiT,
    SD3TextEncoderWithMask,
)

from .flux_modules import (
    PyramidFluxTransformer,
    FluxTextEncoderWithMask,
)
    
from .videollama import (
    videollama_best_of_N,
    videollama_judging
)

from .visionreward import(
    best_of_N,
    judging,
)

def extract_frames(video_path, output_dir=None, target_frame='last', save_frames=False):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save extracted frames
        target_frame (str or int): 'last' for last frame, or frame number to extract
        save_frames (bool): Whether to save all frames to disk
        
    Returns:
        PIL.Image: The requested frame in PIL format
    """
    # If saving frames, create output directory if it doesn't exist
    if save_frames and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert 'last' to actual frame number
    if target_frame == 'last':
        target_frame = total_frames - 1

    frame_count = 0
    target_frame_image = None

    while True:
        ret, frame = cap.read()
        # When no frame is returned, we have reached the end of the video
        if not ret:
            break
        
        # Save frame if requested
        if save_frames and output_dir:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

        # Store target frame
        if frame_count == target_frame:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            target_frame_image = Image.fromarray(frame_rgb)
            
            # If we're not saving all frames, we can break here
            if not save_frames:
                break

        frame_count += 1

    cap.release()
    
    if save_frames:
        print(f"Extraction complete. {frame_count} frames extracted to '{output_dir}'.")
    
    return target_frame_image

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def build_pyramid_dit(
    model_name : str,
    model_path : str,
    torch_dtype,
    use_flash_attn : bool,
    use_mixed_training: bool,
    interp_condition_pos: bool = True,
    use_gradient_checkpointing: bool = False,
    use_temporal_causal: bool = True,
    gradient_checkpointing_ratio: float = 0.6,
):
    model_dtype = torch.float32 if use_mixed_training else torch_dtype
    if model_name == "pyramid_flux":
        dit = PyramidFluxTransformer.from_pretrained(
            model_path, torch_dtype=model_dtype,
            use_gradient_checkpointing=use_gradient_checkpointing, 
            gradient_checkpointing_ratio=gradient_checkpointing_ratio,
            use_flash_attn=use_flash_attn, use_temporal_causal=use_temporal_causal,
            interp_condition_pos=interp_condition_pos, axes_dims_rope=[16, 24, 24],
        )
    elif model_name == "pyramid_mmdit":
        dit = PyramidDiffusionMMDiT.from_pretrained(
            model_path, torch_dtype=model_dtype, use_gradient_checkpointing=use_gradient_checkpointing, 
            gradient_checkpointing_ratio=gradient_checkpointing_ratio,
            use_flash_attn=use_flash_attn, use_t5_mask=True, 
            add_temp_pos_embed=True, temp_pos_embed_type='rope', 
            use_temporal_causal=use_temporal_causal, interp_condition_pos=interp_condition_pos,
        )
    else:
        raise NotImplementedError(f"Unsupported DiT architecture, please set the model_name to `pyramid_flux` or `pyramid_mmdit`")

    return dit


def build_text_encoder(
    model_name : str,
    model_path : str,
    torch_dtype,
    load_text_encoder: bool = True,
):
    # The text encoder
    if load_text_encoder:
        if model_name == "pyramid_flux":
            text_encoder = FluxTextEncoderWithMask(model_path, torch_dtype=torch_dtype)
        elif model_name == "pyramid_mmdit":
            text_encoder = SD3TextEncoderWithMask(model_path, torch_dtype=torch_dtype)
        else:
            raise NotImplementedError(f"Unsupported Text Encoder architecture, please set the model_name to `pyramid_flux` or `pyramid_mmdit`")
    else:
        text_encoder = None

    return text_encoder

def split_prompt(prompt):
    """
        Input: a general prompt
        Output: a prompt list with 3 stages [..., ..., ...]
    """
    # Load model and tokenizer
    # model_name = "/home/lff/bigdata1/huggingface/Meta-Llama-3-8B-Instruct"
    model_name = "/home/lff/bigdata1/huggingface/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto")

    # System prompt
    system_prompt = """Split video generation prompts into 3 stages:
1. Static scene description. Describe all the objects or characters in the input prompt.
2. Action/motion directions. Describe the motion or action of all the objects or characters in the input prompt.
3. Expected ending state. Describe the ending scene of the video.
Provide a description for 3 stages formatted in the same way as the example below:
Input: a girl pushing the chair while her sister is on the chair
Output: [
    "A girl standing behind a chair with her sister sitting on it in a living room, realistic lighting",
    "Smooth pushing motion showing chair movement, sister holding armrests, hair movement",
    "Chair positioned several feet away from original spot, sister smiling while sitting securely"
]
Now process this input:"""

    # Generate answer
    inputs = tokenizer(f"{system_prompt} {prompt}. Remember that you must follow the example format and output only a list.", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=4096)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract result list
    print(result)
    start_idx = result.rfind("[")
    end_idx = result.rfind("]") + 1
    return eval(result[start_idx:end_idx])


class ImageSelector:
    def __init__(self, model_name="llava_qwen", pretrained="lmms-lab/llava-onevision-qwen2-7b-ov", device="cuda", device_map="auto"):
        # Load model
        llava_model_args = {"multimodal": True, "attn_implementation": None}

        overwrite_config = {"image_aspect_ratio": "pad"}
        llava_model_args["overwrite_config"] = overwrite_config
        
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, **llava_model_args
        )
        self.device = device
        self.model.eval()

    def orm(self, prompt, image_file):
        if not os.path.isfile(image_file) or not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            raise ValueError(f"Invalid image file: {image_file}")  

        # Load the image
        image = Image.open(image_file)

        # Process the image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        question = (f"{DEFAULT_IMAGE_TOKEN} This image is generated by a prompt: {prompt[0]}. Does this image accurately represent the prompt? Please answer yes or no without explanation.")

        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

            scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            first_token_prob = scores[0, 0] 
            yes_prob = first_token_prob[yes_token_id].item()
              
        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries. Reponse:" + cur_reponse)
            return False, 0.

        return (cur_reponse == 'yes'), yes_prob

    def parm(self, prompt, image_file, clear):
        if not os.path.isfile(image_file) or not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            raise ValueError(f"Invalid image file: {image_file}")  

        # Load the image
        image = Image.open(image_file)

        # Process the image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        # judge
        if clear == False and self.clarity_judgement(prompt, image_tensor, image) == 'no':
            return False, 1., False 
            # return True, 1., True 
        else:
            potential, no_prob = self.potential_assessment(prompt, image_tensor, image)
            # print(potential, no_prob)
            # potential: True or False, no_prob: probability of no, clear: True or False
            return potential, no_prob, True  
            # potential, no_score, clear
    
    def clarity_judgement(self, prompt, image_tensor, image):
        # Prepare the question
        question = (f"{DEFAULT_IMAGE_TOKEN} This image is a certain step in the text-to-image generation process with a prompt: {prompt[0]}. It is not the final generated one, and will keep iterating better. Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt? (The image, which needn't to be confused but can be clear and basically judged the object, can be used to judge the potential) Answer yes or no without explanation.")
        question = f"""{DEFAULT_IMAGE_TOKEN} \n The above image is a certain step  in the text to image generation process. This image is not the final generated one, and will keep iterating better. The image is generated by prompt '{prompt[0]}.'
        Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt?(The image, which needn't to be confused  but can be clear and basically judged the object, can be used to judge the potential)
        Answer yes or no with no explanation.(yes or no in lowercase and no full point)"""
        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries.")

        return cur_reponse

    def potential_assessment(self, prompt, image_tensor, image):
        # Prepare the question
        clear_question= f"""{DEFAULT_IMAGE_TOKEN} \n The above image is a certain step  in the text to image generation process. This image is not the final generated one, and will keep iterating better. The image is generated by prompt '{prompt[0]}.'
        Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt?(The image, which needn't to be confused  but can be clear and basically judged the object, can be used to judge the potential)
        Answer yes or no with no explanation.(yes or no in lowercase and no full point)"""
        question = (f"{DEFAULT_IMAGE_TOKEN} Do you think whether the image has the potential to iterate to the image satisfied the prompt? Please answer yes or no without explanation.")
        question='Do you think whether the image has the potential to iterate to the image satisfied the prompt? Please answer yes or no with no explanation.(yes or no in lowercase and no full point)'
        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], clear_question)
        conv.append_message(conv.roles[1], 'yes')
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

            scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            first_token_prob = scores[0, 0] 
            no_prob = first_token_prob[no_token_id].item()
              
        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries.")
            return False, 50.

        return ('yes' in cur_reponse), no_prob

class PyramidDiTForVideoGeneration:
    """
        The pyramid dit for both image and video generation, The running class wrapper
        This class is mainly for fixed unit implementation: 1 + n + n + n
    """
    def __init__(self, model_path, model_dtype='bf16', model_name='pyramid_mmdit', use_gradient_checkpointing=False, 
        return_log=True, model_variant="diffusion_transformer_768p", timestep_shift=1.0, stage_range=[0, 1/3, 2/3, 1],
        sample_ratios=[1, 1, 1], scheduler_gamma=1/3, use_mixed_training=False, use_flash_attn=False, 
        load_text_encoder=True, load_vae=True, max_temporal_length=31, frame_per_unit=1, use_temporal_causal=True, 
        corrupt_ratio=1/3, interp_condition_pos=True, stages=[1, 2, 4], video_sync_group=8, gradient_checkpointing_ratio=0.6, **kwargs,
    ):
        super().__init__()

        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.stages = stages
        self.sample_ratios = sample_ratios
        self.corrupt_ratio = corrupt_ratio

        dit_path = os.path.join(model_path, model_variant)

        # The dit
        self.dit = build_pyramid_dit(
            model_name, dit_path, torch_dtype, 
            use_flash_attn=use_flash_attn, use_mixed_training=use_mixed_training,
            interp_condition_pos=interp_condition_pos, use_gradient_checkpointing=use_gradient_checkpointing,
            use_temporal_causal=use_temporal_causal, gradient_checkpointing_ratio=gradient_checkpointing_ratio,
        )

        # The text encoder
        self.text_encoder = build_text_encoder(
            model_name, model_path, torch_dtype, load_text_encoder=load_text_encoder,
        )
        self.load_text_encoder = load_text_encoder

        # The base video vae decoder
        if load_vae:
            self.vae = CausalVideoVAE.from_pretrained(os.path.join(model_path, 'causal_video_vae'), torch_dtype=torch_dtype, interpolate=False)
            # Freeze vae
            for parameter in self.vae.parameters():
                parameter.requires_grad = False
        else:
            self.vae = None
        self.load_vae = load_vae
        
        # For the image latent
        if model_name == "pyramid_flux":
            self.vae_shift_factor = -0.04
            self.vae_scale_factor = 1 / 1.8726
        elif model_name == "pyramid_mmdit":
            self.vae_shift_factor = 0.1490
            self.vae_scale_factor = 1 / 1.8415
        else:
            raise NotImplementedError(f"Unsupported model name : {model_name}")

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.downsample = 8

        # Configure the video training hyper-parameters
        # The video sequence: one frame + N * unit
        self.frame_per_unit = frame_per_unit
        self.max_temporal_length = max_temporal_length
        assert (max_temporal_length - 1) % frame_per_unit == 0, "The frame number should be divided by the frame number per unit"
        self.num_units_per_video = 1 + ((max_temporal_length - 1) // frame_per_unit) + int(sum(sample_ratios))

        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            shift=timestep_shift, stages=len(self.stages), 
            stage_range=stage_range, gamma=scheduler_gamma,
        )
        print(f"The start sigmas and end sigmas of each stage is Start: {self.scheduler.start_sigmas}, End: {self.scheduler.end_sigmas}, Ori_start: {self.scheduler.ori_start_sigmas}")
        
        self.cfg_rate = 0.1
        self.return_log = return_log
        self.use_flash_attn = use_flash_attn
        self.model_name = model_name
        self.sequential_offload_enabled = False
        self.accumulate_steps = 0
        self.video_sync_group = video_sync_group
        
    def _enable_sequential_cpu_offload(self, model):
        self.sequential_offload_enabled = True
        torch_device = torch.device("cuda")
        device_type = torch_device.type
        device = torch.device(f"{device_type}:0")
        offload_buffers = len(model._parameters) > 0
        cpu_offload(model, device, offload_buffers=offload_buffers)
    
    def enable_sequential_cpu_offload(self):
        self._enable_sequential_cpu_offload(self.text_encoder)
        self._enable_sequential_cpu_offload(self.dit)

    def load_checkpoint(self, checkpoint_path, model_key='model', **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dit_checkpoint = OrderedDict()
        for key in checkpoint:
            if key.startswith('vae') or key.startswith('text_encoder'):
                continue
            if key.startswith('dit'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                dit_checkpoint[new_key] = checkpoint[key]
            else:
                dit_checkpoint[key] = checkpoint[key]

        load_result = self.dit.load_state_dict(dit_checkpoint, strict=True)
        print(f"Load checkpoint from {checkpoint_path}, load result: {load_result}")

    def load_vae_checkpoint(self, vae_checkpoint_path, model_key='model'):
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        checkpoint = checkpoint[model_key]
        loaded_checkpoint = OrderedDict()
        
        for key in checkpoint.keys():
            if key.startswith('vae.'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                loaded_checkpoint[new_key] = checkpoint[key]

        load_result = self.vae.load_state_dict(loaded_checkpoint)
        print(f"Load the VAE from {vae_checkpoint_path}, load result: {load_result}")

    @torch.no_grad()
    def add_pyramid_noise(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        """
        add the noise for each pyramidal stage
            noting that, this method is a general strategy for pyramid-flow, it 
            can be used for both image and video training.
            You can also use this method to train pyramid-flow with full-sequence 
            diffusion in video generation (without using temporal pyramid and autoregressive modeling)

        Params:
            latent_list: [low_res, mid_res, high_res] The vae latents of all stages
            sample_ratios: The proportion of each stage in the training batch
        """
        noise = torch.randn_like(latents_list[-1])
        device = noise.device
        dtype = latents_list[-1].dtype
        t = noise.shape[2]

        stages = len(self.stages)
        tot_samples = noise.shape[0]
        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)
        
        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)

        noise_list = list(reversed(noise_list))   # make sure from low res to high res
        
        # To calculate the padding batchsize and column size
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))
        
        column_to_stage = {}
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num

        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        # from low resolution to high resolution
        for index in range(column_size):
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]
            
            if i_s == 0:
                start_point = noise_list[i_s][index::column_size]
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            # To sample a timestep
            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            # interpolate the latent
            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * self.corrupt_ratio

            # [stage1_latent, stage2_latent, ..., stagen_latent], which will be concat after patching
            noisy_latents_list.append([noisy_latents.to(dtype)])
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(start_point - end_point)     # The standard rectified flow matching objective

        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    def sample_stage_length(self, num_stages, max_units=None):
        max_units_in_training = 1 + ((self.max_temporal_length - 1) // self.frame_per_unit)
        cur_rank = get_rank()

        self.accumulate_steps = self.accumulate_steps + 1
        total_turns =  max_units_in_training // self.video_sync_group
        update_turn = self.accumulate_steps % total_turns

        # # uniformly sampling each position
        cur_highres_unit = max(int((cur_rank % self.video_sync_group + 1) + update_turn * self.video_sync_group), 1)
        cur_mid_res_unit = max(1 + max_units_in_training - cur_highres_unit, 1)
        cur_low_res_unit = cur_mid_res_unit

        if max_units is not None:
            cur_highres_unit = min(cur_highres_unit, max_units)
            cur_mid_res_unit = min(cur_mid_res_unit, max_units)
            cur_low_res_unit = min(cur_low_res_unit, max_units)

        length_list = [cur_low_res_unit, cur_mid_res_unit, cur_highres_unit]
        
        assert len(length_list) == num_stages

        return length_list

    @torch.no_grad()
    def add_pyramid_noise_with_temporal_pyramid(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        """
        add the noise for each pyramidal stage, used for AR video training with temporal pyramid
        Params:
            latent_list: [low_res, mid_res, high_res] The vae latents of all stages
            sample_ratios: The proportion of each stage in the training batch
        """
        stages = len(self.stages)
        tot_samples = latents_list[0].shape[0]
        device = latents_list[0].device
        dtype = latents_list[0].dtype

        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)

        noise = torch.randn_like(latents_list[-1])
        t = noise.shape[2]

        # To allocate the temporal length of each stage, ensuring the sum == constant
        max_units = 1 + (t - 1) // self.frame_per_unit

        if is_sequence_parallel_initialized():
            max_units_per_sample = torch.LongTensor([max_units]).to(device)
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            max_units_per_sample = all_to_all(max_units_per_sample.unsqueeze(1).repeat(1, sp_group_size), sp_group, sp_group_size, scatter_dim=1, gather_dim=0).squeeze(1)
            max_units = min(max_units_per_sample.cpu().tolist())

        num_units_per_stage = self.sample_stage_length(stages, max_units=max_units)   # [The unit number of each stage]

        # we needs to sync the length alloc of each sequence parallel group
        if is_sequence_parallel_initialized():
            num_units_per_stage = torch.LongTensor(num_units_per_stage).to(device)
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(num_units_per_stage, global_src_rank, group=get_sequence_parallel_group())
            num_units_per_stage = num_units_per_stage.tolist()

        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)

        noise_list = list(reversed(noise_list))   # make sure from low res to high res

        # To calculate the batchsize and column size
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))

        column_to_stage = {}
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num

        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        # from low resolution to high resolution
        for index in range(column_size):
            # First prepare the trainable latent construction
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]

            if i_s == 0:
                start_point = noise_list[i_s][index::column_size]
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            # To sample a timestep
            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)
            noise_ratios = ratios * start_sigma + (1 - ratios) * end_sigma

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            # interpolate the latent
            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            # The flow matching object
            target_latents = start_point - end_point

            # pad the noisy previous
            num_units = num_units_per_stage[i_s]
            num_units = min(num_units, 1 + (t - 1) // self.frame_per_unit)
            actual_frames = 1 + (num_units - 1) * self.frame_per_unit

            noisy_latents = noisy_latents[:, :, :actual_frames]
            target_latents = target_latents[:, :, :actual_frames]

            clean_latent = clean_latent[:, :, :actual_frames]
            stage_noise = noise_list[i_s][index::column_size][:, :, :actual_frames]

            # only the last latent takes part in training
            noisy_latents = noisy_latents[:, :, -self.frame_per_unit:] 
            target_latents = target_latents[:, :, -self.frame_per_unit:]

            last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * self.corrupt_ratio

            if num_units == 1:
                stage_input = [noisy_latents.to(dtype)]
            else:
                # add the random noise for the last cond clip
                last_cond_latent = clean_latent[:, :, -(2*self.frame_per_unit):-self.frame_per_unit]

                while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                    last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)

                # We adding some noise to corrupt the clean condition
                last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent

                # concat the corrupted condition and the input noisy latents
                stage_input = [noisy_latents.to(dtype), last_cond_latent.to(dtype)]

                cur_unit_num = 2
                cur_stage = i_s

                while cur_unit_num < num_units:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_num += 1
                    cond_latents = latents_list[cur_stage][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, -(cur_unit_num * self.frame_per_unit) : -((cur_unit_num - 1) * self.frame_per_unit)]
                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))

                if cur_stage == 0 and cur_unit_num < num_units:
                    cond_latents = latents_list[0][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, :-(cur_unit_num * self.frame_per_unit)]

                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))

            stage_input = list(reversed(stage_input))
            noisy_latents_list.append(stage_input)
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(target_latents)     # The standard rectified flow matching objective
        
        return noisy_latents_list, ratios_list, timesteps_list, targets_list
    
    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num):
        # x is the origin vae latent
        vae_latent_list = []
        vae_latent_list.append(x)

        temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        for _ in range(stage_num):
            height //= 2
            width //= 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)

        vae_latent_list = list(reversed(vae_latent_list))
        return vae_latent_list

    @torch.no_grad()
    def get_vae_latent(self, video, use_temporal_pyramid=True):
        if self.load_vae:
            assert video.shape[1] == 3, "The vae is loaded, the input should be raw pixels"
            video = self.vae.encode(video).latent_dist.sample() # [b c t h w]

        if video.shape[2] == 1:
            # is image
            video = (video - self.vae_shift_factor) * self.vae_scale_factor
        else:
            # is video
            video[:, :, :1] = (video[:, :, :1] - self.vae_shift_factor) * self.vae_scale_factor
            video[:, :, 1:] =  (video[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        
        # Get the pyramidal stages
        vae_latent_list = self.get_pyramid_latent(video, len(self.stages) - 1)

        if use_temporal_pyramid:
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise_with_temporal_pyramid(vae_latent_list, self.sample_ratios)
        else:
            # Only use the spatial pyramidal (without temporal ar)
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise(vae_latent_list, self.sample_ratios)
        
        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    @torch.no_grad()
    def get_text_embeddings(self, text, rand_idx, device):
        if self.load_text_encoder:
            batch_size = len(text)   # Text is a str list
            for idx in range(batch_size):
                if rand_idx[idx].item():
                    text[idx] = ''
            return self.text_encoder(text, device)   # [b s c]
        else:
            batch_size = len(text['prompt_embeds'])

            for idx in range(batch_size):
                if rand_idx[idx].item():
                    text['prompt_embeds'][idx] = self.null_text_embeds['prompt_embed'].to(device)
                    text['prompt_attention_mask'][idx] = self.null_text_embeds['prompt_attention_mask'].to(device)
                    text['pooled_prompt_embeds'][idx] = self.null_text_embeds['pooled_prompt_embed'].to(device)

            return text['prompt_embeds'], text['prompt_attention_mask'], text['pooled_prompt_embeds']

    def calculate_loss(self, model_preds_list, targets_list):
        loss_list = []
    
        for model_pred, target in zip(model_preds_list, targets_list):
            # Compute the loss.
            loss_weight = torch.ones_like(target)

            loss = torch.mean(
                (loss_weight.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss_list.append(loss)

        diffusion_loss = torch.cat(loss_list, dim=0).mean()

        if self.return_log:
            log = {}
            split="train"
            log[f'{split}/loss'] = diffusion_loss.detach()
            return diffusion_loss, log
        else:
            return diffusion_loss, {}

    def __call__(self, video, text, identifier=['video'], use_temporal_pyramid=True, accelerator: Accelerator=None):
        xdim = video.ndim
        device = video.device

        if 'video' in identifier:
            assert 'image' not in identifier
            is_image = False
        else:
            assert 'video' not in identifier
            video = video.unsqueeze(2)  # 'b c h w -> b c 1 h w'
            is_image = True

        # TODO: now have 3 stages, firstly get the vae latents
        with torch.no_grad(), accelerator.autocast():
            # 10% prob drop the text
            batch_size = len(video)
            rand_idx = torch.rand((batch_size,)) <= self.cfg_rate
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.get_text_embeddings(text, rand_idx, device)
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.get_vae_latent(video, use_temporal_pyramid=use_temporal_pyramid)

        timesteps = torch.cat([timestep.unsqueeze(-1) for timestep in timesteps_list], dim=-1)
        timesteps = timesteps.reshape(-1)

        assert timesteps.shape[0] == prompt_embeds.shape[0]

        # DiT forward
        model_preds_list = self.dit(
            sample=noisy_latents_list,
            timestep_ratio=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
        )

        # calculate the loss
        return self.calculate_loss(model_preds_list, targets_list)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        temp,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(temp),
            int(height) // self.downsample,
            int(width) // self.downsample,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def sample_block_noise(self, bs, ch, temp, height, width):
        gamma = self.scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma)
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise

    @torch.no_grad()
    def generate_one_unit_parm(
        self,
        latents,
        past_conditions, # List of past conditions, contains the conditions of each stage
        prompt,
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        temp,
        device,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        is_first_frame: bool = False,
    ):
        # PARM init
        clear = False
        filtered = False
        score = 1.
        stages = self.stages
        intermed_latents = []

        for i_s in range(len(stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise    # To fix the block artifact

            for idx, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                if is_sequence_parallel_initialized():
                    # sync the input latent
                    sp_group_rank = get_sequence_parallel_group_rank()
                    global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
                    torch.distributed.broadcast(latent_model_input, global_src_rank, group=get_sequence_parallel_group())
                
                latent_model_input = past_conditions[i_s] + [latent_model_input]

                noise_pred = self.dit(
                    sample=[latent_model_input],
                    timestep_ratio=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                )

                noise_pred = noise_pred[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if is_first_frame:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)
            # PARM
            if clear: 
                pass
            else:
                image_filename = "waiting_for_judge.png"
                image_path = os.path.join('./', image_filename)
                generated_latents = torch.cat([latents], dim=2)
                images_list = self.decode_latent(generated_latents, save_memory=True, inference_multigpu=False)
                for image in images_list:
                    image.save(image_path)
                # filtered: potential or not 
                filtered, score, clear = self.selector.parm([prompt], image_path, clear=clear)
        
        return intermed_latents, filtered, score

    @torch.no_grad()
    def generate_one_unit(
        self,
        latents,
        past_conditions, # List of past conditions, contains the conditions of each stage
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        temp,
        device,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        is_first_frame: bool = False,
    ):
        stages = self.stages
        intermed_latents = []

        for i_s in range(len(stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise    # To fix the block artifact

            for idx, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                if is_sequence_parallel_initialized():
                    # sync the input latent
                    sp_group_rank = get_sequence_parallel_group_rank()
                    global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
                    torch.distributed.broadcast(latent_model_input, global_src_rank, group=get_sequence_parallel_group())
                
                latent_model_input = past_conditions[i_s] + [latent_model_input]

                noise_pred = self.dit(
                    sample=[latent_model_input],
                    timestep_ratio=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                )

                noise_pred = noise_pred[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if is_first_frame:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)

        return intermed_latents

    @torch.no_grad()
    def generate_i2v(
        self,
        prompt: Union[str, List[str]] = '',
        input_image: PIL.Image = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 4.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()

        width = input_image.width
        height = input_image.height

        assert temp % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)

        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        expanded_prompt = []

        for p in prompt:
            expanded_prompt.append(p*num_images_per_prompt)
        
        prompt = expanded_prompt
        batch_size = len(prompt)

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)
        
        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda") 
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
            self.vae.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp+1)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by defalut, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = temp // self.frame_per_unit
        stages = self.stages

        # encode the image latents
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        input_image_tensor = image_transform(input_image).unsqueeze(0).unsqueeze(2)   # [b c 1 h w]
        input_image_latent = (self.vae.encode(input_image_tensor.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample() - self.vae_shift_factor) * self.vae_scale_factor  # [b c 1 h w]

        if is_sequence_parallel_initialized():
            # sync the image latent across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(input_image_latent, global_src_rank, group=get_sequence_parallel_group())

        generated_latents_list = [input_image_latent]    # The generated results
        last_generated_latents = input_image_latent

        if cpu_offloading:
            self.vae.to("cpu")
            if not self.sequential_offload_enabled:
                self.dit.to("cuda")
            torch.cuda.empty_cache()
        
        for unit_index in tqdm(range(1, num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
        
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            # prepare the condition latents
            past_condition_latents = []
            clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
            
            for i_s in range(len(stages)):
                last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]

                stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
        
                # pad the past clean latents
                cur_unit_num = unit_index
                cur_stage = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
            
                stage_input = list(reversed(stage_input))
                past_condition_latents.append(stage_input)

            intermed_latents = self.generate_one_unit(
                latents[:,:,(unit_index - 1) * self.frame_per_unit:unit_index * self.frame_per_unit],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                height,
                width,
                self.frame_per_unit,
                device,
                dtype,
                generator,
                is_first_frame=False,
            )
    
            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image



    @torch.no_grad()
    def branch_step_generate(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
        num_paths: Optional[int] = 5,
        num_branches: Optional[int] = 1,
        output_dir: Optional[str] = None,
        select: bool = False,
        save_videos: bool = False,
    ):
        """
        step_generate performs interactive process control during video generation.
        For each generation unit (frame or group of frames), multiple candidate latent outputs
        are generated using different random seeds. The user is then prompted to select the desired candidate.
        The chosen latent is used as the conditioning for generating the next unit.
        """
        # Create a directory to save candidate frames.
        if output_dir is None:
            output_dir = "./branch_step_generate"

        title = "videotemp"
        current_time = datetime.datetime.now()
        timestamp_dir = current_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, title, timestamp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # The text prompt processing is similar to generate.
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)
        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        batch_size = len(prompt)

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif negative_prompt is None:
            negative_prompt = [""]
        else:
            assert isinstance(negative_prompt, list)


        # Get text embeddings.
        device = self.device if not cpu_offloading else torch.device("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        
        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)


        # Create initial latent noise.
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else self.dit.config.in_channels

        if generator is None:
            generators = [torch.Generator(device=self.device).manual_seed(i * 5000) for i in range(num_paths)]
        else:
            base_seed = generator.initial_seed()
            generators = [generator] * num_paths
        
        latents_list = []
        for gen in generators:
            torch.cuda.empty_cache()
            latent = self.prepare_latents(batch_size, num_channels_latents, temp, height, width, prompt_embeds.dtype, device, gen)
            latents_list.append(latent)
        
        processed_latents_list = []
        for latent in latents_list:
            temp, height, width = latent.shape[-3], latent.shape[-2], latent.shape[-1]
            current_latent = rearrange(latent, 'b c t h w -> (b t) c h w')
    
            # Pyramid downsampling
            current_height, current_width = height, width
            for _ in range(len(self.stages) - 1):
                current_height //= 2
                current_width //= 2
                current_latent = F.interpolate(current_latent, size=(current_height, current_width), mode='bilinear') * 2
                torch.cuda.empty_cache()

            current_latent = rearrange(current_latent, '(b t) c h w -> b c t h w', t=temp)
            processed_latents_list.append(current_latent)

        # Calculate number of units (each unit generates self.frame_per_unit frames).
        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages
        generated_latents_lists =  [[] for _ in range(num_paths)]  # final chosen latent outputs for each unit
        candidate_idx = [i for i in range(num_paths)]

        # For each generation unit, produce candidate outputs and choose one interactively.
        for unit_index in range(num_units):
            
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n--- Generating Unit {unit_index} ---")
            
            # candidate_info(dict) consists lists of branch outputs
            candidates = []
            candidate_images = []
            candidate_info = {i: [] for i in candidate_idx}

            for candidate_index in candidate_idx:
                torch.cuda.empty_cache()

                for branch in range(num_branches):
                    # Vary the generator per branch
                    branch_gen = generators[candidate_index]

                    if num_branches > 1:
                        branch_seed = branch_gen.initial_seed() + branch * 17
                        branch_gen.manual_seed(branch_seed)

                    if unit_index == 0:
                        past_condition_latents = [[] for _ in range(len(stages))]
                        intermed_latents = self.generate_one_unit(
                            processed_latents_list[candidate_index][:, :, :1],
                            past_condition_latents,
                            prompt_embeds,
                            prompt_attention_mask,
                            pooled_prompt_embeds,
                            num_inference_steps,
                            current_height,
                            current_width,
                            1,
                            device,
                            self.dtype,
                            branch_gen,
                            is_first_frame=True,
                        )

                        branch_latent = intermed_latents[-1]
                        branch_images = self.decode_latent(branch_latent, save_memory=save_memory, inference_multigpu=inference_multigpu)
                        # Assume candidate_images is a list; select the first image for display.
                        branch_image = branch_images[0] if isinstance(branch_images, list) else branch_images
                        info = {
                            "candidate_index": candidate_index, 
                            "branch": branch, 
                            "latent":branch_latent,
                            "image":branch_image
                        }
                        # Save image to file.
                        save_path = os.path.join(output_dir, f"unit_{unit_index}_candidate_{candidate_index}_branch_{branch}.png")
                        branch_image.save(save_path)
                        print(f"Saved candidate image: {save_path}")

                    else:
                        past_condition_latents = []

                        if len(generated_latents_lists[candidate_index]) > 0:
                            previous_latents = torch.cat(generated_latents_lists[candidate_index], dim=2)
                            clean_latents_list = self.get_pyramid_latent(previous_latents, len(stages) - 1)
                    
                            for i_s in range(len(stages)):
                                last_cond_latent = clean_latents_list[i_s][:, :, -(self.frame_per_unit):]
                                stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                            
                                cur_unit_num = unit_index
                                cur_stage = i_s
                                cur_unit_ptx = 1
                            
                                while cur_unit_ptx < cur_unit_num:
                                    cur_stage = max(cur_stage - 1, 0)
                                    if cur_stage == 0:
                                        break
                                    cur_unit_ptx += 1
                                    cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit): -((cur_unit_ptx - 1) * self.frame_per_unit)]
                                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                                    cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                                stage_input = list(reversed(stage_input))
                                past_condition_latents.append(stage_input)

                        else:
                            past_condition_latents = [[] for _ in range(len(stages))]

                        intermed_latents = self.generate_one_unit(
                            processed_latents_list[candidate_index][:, :, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                            past_condition_latents,
                            prompt_embeds,
                            prompt_attention_mask,
                            pooled_prompt_embeds,
                            video_num_inference_steps,
                            current_height,
                            current_width,
                            self.frame_per_unit,
                            device,
                            self.dtype,
                            branch_gen,
                            is_first_frame=False,
                        )

                        branch_latent = intermed_latents[-1]
                
                        info = {
                            "candidate_index": candidate_index,
                            "branch":branch,
                            "latent":branch_latent
                        }

                        print(f"Unit_{unit_index}: Saved candidate_{candidate_index} info.")
                    
                    candidate_info[candidate_index].append(info)
                    torch.cuda.empty_cache()

           
            candidate_branch_histories: Dict[int, List[List[torch.Tensor]]] = {}
            # 外层循环用于更新candidate的history
            for candidate_index in candidate_idx:
                branch_histories = [] # List: 包含所有branch的历史
                base_history = generated_latents_lists[candidate_index].copy() #candidate history
                
                # 内层循环用于得到branch的history
                for branch_info in candidate_info[candidate_index]:
                    history = base_history.copy()
                    history.append(branch_info["latent"])
                    branch_histories.append(history)
                
                candidate_branch_histories[candidate_index] = branch_histories

            if select:
                selections = self.branch_plot_and_select(candidate_idx, candidate_info, candidate_branch_histories, 
                                                        unit_index, inference_multigpu=inference_multigpu,
                                                        base_dir = output_dir)

                for candidate_index in candidate_idx:
                    chosen_branch = selections.get(candidate_index, -1)
                    if chosen_branch != -1:
                        chosen_history = candidate_branch_histories[candidate_index][chosen_branch]
                        generated_latents_lists[candidate_index] = chosen_history

            else:
                for candidate_index in candidate_idx:
                    chosen_branch = 0
                    chosen_history = candidate_branch_histories[candidate_index][0]
                    generated_latents_lists[candidate_index] = chosen_history

        for idx, latent_list in enumerate(generated_latents_lists):
            if len(latent_list) == num_units:
                generated_latents = torch.cat(latent_list, dim=2)
                image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
                output_path = os.path.join(output_dir, f"default_setting_candidate_{idx}.mp4")
                export_to_video(image, output_path, fps=24)

        return image

    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()


        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)

        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        expanded_prompt = []

        for p in prompt:
            expanded_prompt.append(p * num_images_per_prompt)
        
        prompt = expanded_prompt
        batch_size = len(prompt)


        negative_prompt = negative_prompt or ""

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        else:
            assert isinstance(negative_prompt, list)

        expanded_prompt = []

        for p in negative_prompt:
            expanded_prompt.append(p*num_images_per_prompt)
        
        negative_prompt = expanded_prompt


        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)


        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
                self.dit.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            # guidance_scale_list = torch.linspace(max_guidance_scale, min_guidance_scale, temp).tolist()
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        
        # Loop multiple downscaling operation
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        # Each unit generated at once
        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
            
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents = self.generate_one_unit(
                    latents[:,:,:1],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )


            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                # Intermediate latents
                intermed_latents = self.generate_one_unit(
                    latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    self.frame_per_unit,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)
        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image






        




    def decode_latent(self, latents, save_memory=True, inference_multigpu=False, numpy_output=False, force_video=False):
        # only the main process needs vae decoding
        if inference_multigpu and get_rank() != 0:
            return None

        if latents.shape[2] == 1 and not force_video:
            # print(f"Decode Image")
            latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        elif latents.shape[2] > 1:
            print(f"Decode Video Sequence")
            latents[:, :, :1] = (latents[:, :, :1] / self.vae_scale_factor) + self.vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / self.vae_video_scale_factor) + self.vae_video_shift_factor


        if save_memory:
            # reducing the tile size and temporal chunk window size
            image = self.vae.decode(latents, temporal_chunk=True, window_size=1, tile_sample_min_size=256).sample
        else:
            image = self.vae.decode(latents, temporal_chunk=True, window_size=2, tile_sample_min_size=512).sample

        image = image.mul(127.5).add(127.5).clamp(0, 255).byte()
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().numpy()
        if numpy_output:
            return image
        image = self.numpy_to_pil(image)
        
        return image



    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @property
    def device(self):
        return next(self.dit.parameters()).device

    @property
    def dtype(self):
        return next(self.dit.parameters()).dtype

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0


    @torch.no_grad()
    def video_tree_generate(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]] = "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False,
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
        output_dir: Optional[str] = None,
        num_branch: List[int] = None,
        pa: bool = False,
    ):
        # Upd: 增强初始Generator的随机性
        """
        Generate a video tree according to requirements
        Using a queue to store videos' encoding, i.e. video generation process
        Video Encoding Method:
            010324102
            - number[i]: branch index at unit i
        Args:
            num_branch: List[int], number of branches at each step
            pa: Bool, Process_Available(True=Save Process Videos, False=No Save Process)
        """
        from queue import SimpleQueue
        
        if output_dir is None:
            output_dir = "./video_tree_generate"
        current_time = datetime.datetime.now()
        timestamp_dir = current_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Input validation
        if num_branch is None or len(num_branch) == 0:
            raise ValueError("num_branch must be provided and non-empty")

        # Text prompt processing
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)

        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        expanded_prompt = []

        for p in prompt:
            expanded_prompt.append(p * num_branch[0])
        
        prompt = expanded_prompt
        batch_size = len(prompt)


        negative_prompt = negative_prompt or ""

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        else:
            assert isinstance(negative_prompt, list)

        expanded_prompt = []

        for p in negative_prompt:
            expanded_prompt.append(p*num_branch[0])
        
        negative_prompt = expanded_prompt

        # Get text embeddings
        device = self.device if not cpu_offloading else torch.device("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if use_linear_guidance:
            self._guidance_scale = guidance_scale_list[unit_index]
            self._video_guidance_scale = guidance_scale_list[unit_index]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Initialize queue and dictionary
        encoding_queue = SimpleQueue()
        latent_dict = {}

        # Setup latent dimensions and prepare initial latents
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else self.dit.config.in_channels

        # Initialize base generator for branch seeds
        base_generator = generator if generator is not None else torch.Generator(device=self.device)
        base_seed = base_generator.initial_seed() if generator is not None else torch.randint(0, 2**63-1, (1,), device=self.device).item()

        # random seeds generate
        perm_size = max(num_branch[0], 2**16)
        nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
        seeds = ((nums[:num_branch[0]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 

        # initial noisy latent list
        noisy_latents = []
        cur_height = height
        cur_width = width


        # Tree depth 0
        for i in range(num_branch[0]):
            
            #print(f"Current Encoding: {i}, seed: {int(seeds[i])}")
            #branch_generator = torch.Generator(device=self.device).manual_seed(int(seeds[i]))
            
            seed = int(torch.randint(0, 2**31-1, (1,), device=self.device))
            branch_generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Current Encoding: {i}, seed: {seed}")

            noisy_latent, cur_height, cur_width = self.prepare_processed_latents(
                1,
                num_channels_latents,
                temp,
                height,
                width,
                prompt_embeds.dtype,
                device,
                branch_generator,
            )

            noisy_latents.append(noisy_latent)


            past_condition_latents = [[] for _ in range(len(self.stages))]
            intermed_latents = self.generate_one_unit(
                noisy_latent[:,:,:1],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                cur_height,
                cur_width,
                1,
                device,
                self.dtype,
                branch_generator,
                is_first_frame=True
            )

            # 直接存储第一帧的latents
            latent_dict[str(i)] = intermed_latents[-1]
            encoding_queue.put(str(i))

            if pa:
                images = self.decode_latent(intermed_latents[-1], save_memory=save_memory, inference_multigpu=inference_multigpu)
                for idx, img in enumerate(images):
                    filename = f"branch_{i}_frame_0_{idx}.png"
                    img.save(os.path.join(output_dir, filename))
                    print(f"Image Saved at {os.path.join(output_dir, filename)}")

        print(f"Cuurent Depth: 0")

        # 处理后续分支时
        while not encoding_queue.empty():

            prefix_encoding = encoding_queue.get()

            current_depth = len(prefix_encoding)

            print(f"Current Depth: {current_depth}")

            if current_depth >= len(num_branch):

                history_latent_list = [] 
                # 分割prefix_encoding依次append前序的latent_dict[str]
                for i in range(current_depth):
                    partial_encoding = prefix_encoding[:i+1]
                    if partial_encoding in latent_dict:
                        history_latent_list.append(latent_dict[partial_encoding])
                
                images = self.decode_latent(torch.cat(history_latent_list,dim=2), save_memory=save_memory, inference_multigpu=inference_multigpu)
                video_path = os.path.join(output_dir, f"final_{prefix_encoding}.mp4")
                export_to_video(images, video_path, fps=24)
                print(f"Save Final Results at: {video_path}")

            else:
                
                # random seeds generate
                perm_size = max(num_branch[current_depth], 2**16)
                nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
                seeds = ((nums[:num_branch[current_depth]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 
                
                for i in range(num_branch[current_depth]):
                    
                    # Obtain history_latents again, in case of getting other frame's history
                    h_latent_list = []
                    for j in range(current_depth):
                        partial_encoding = prefix_encoding[:j+1]
                        if partial_encoding in latent_dict:
                            h_latent_list.append(latent_dict[partial_encoding])

                    current_encoding = prefix_encoding + str(i)
                    print(f"Current Encoding: {current_encoding}, seed: {int(seeds[i])}")
                    branch_generator = torch.Generator(device=self.device).manual_seed(int(seeds[i]))

                    # 准备条件
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(h_latent_list,dim=2), len(self.stages) - 1)
                    
                    for i_s in range(len(self.stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]
                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                        
                        # 添加多级历史帧条件
                        cur_unit_num = current_depth
                        cur_stage = i_s
                        cur_unit_ptx = 1
                        
                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    # 生成新的单元
                    intermed_latents = self.generate_one_unit(
                        noisy_latents[int(prefix_encoding[:1])][:, :, 1 + (current_depth - 1) * self.frame_per_unit:1 + current_depth * self.frame_per_unit],
                        past_condition_latents,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        cur_height,
                        cur_width,
                        self.frame_per_unit,
                        device,
                        self.dtype,
                        branch_generator,
                        is_first_frame=False
                    )

                    # 将历史latent和新生成的latent拼接在一起
                    h_latent_list.append(intermed_latents[-1])
                    combined_latents = torch.cat(h_latent_list, dim=2)  # 在时间维度上拼接
                    latent_dict[current_encoding] = intermed_latents[-1]
                    encoding_queue.put(current_encoding)

                    if pa :
                        # 解码完整的视频序列
                        images = self.decode_latent(combined_latents, 
                                                    save_memory=save_memory, 
                                                    inference_multigpu=inference_multigpu)
                        filename = f"path_{current_encoding}.mp4"
                        video_path = os.path.join(output_dir, filename)
                        export_to_video(images, video_path, fps=24)
                        print(f"Video saved at {os.path.join(output_dir, filename)}")

                        # 如果加入Reward, 在reward加入的depth, 先进行完全的视频解码, 然后reward
                        # Y/N, 用于保留/删除在queue/latent_dict中对应的这个分支

                    torch.cuda.empty_cache()

        # Clean up if using CPU offloading
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.dit.to("cpu")
            torch.cuda.empty_cache()

        return latent_dict



    def prepare_processed_latents(
        self,
        batch_size,
        num_channels_latents,
        temp,
        height,
        width,
        dtype,
        device,
        generator,
    ):

        device = torch.device(device) 
        latents = self.prepare_latents(batch_size, num_channels_latents,
                                            temp,height,width,dtype,device,generator)
        
        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]
        
        # Rearrange and downscale
        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        
        # Loop multiple downscaling operation
        for _ in range(len(self.stages)-1):
            height //= 2
            width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        return latents, height, width

    # 使用VideoLLaMA实现的CoT过程
    @torch.no_grad()
    def vl_cot_gen(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]] = "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False,
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
        output_dir: Optional[str] = None,
        num_branch: List[int] = None,
        reward_model = None,
        reward_stages: List[int] = None,
        processor = None,
    ):
        """Chain-of-Thought Video Generation
            reward_model: vlm-based video judgement model.
            reward_stages: stage(depth) for reward model working.
        
        Output: 
            Videos, as the generation result.
        """



        from queue import SimpleQueue
        
        # 创建输出文件夹
        if output_dir is None:
            output_dir = "./video_tree_generate"
        current_time = datetime.datetime.now()
        timestamp_dir = current_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Input validation
        if num_branch is None or len(num_branch) == 0:
            raise ValueError("num_branch must be provided and non-empty")

        # Text prompt processing
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)

        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        expanded_prompt = []

        for p in prompt:
            expanded_prompt.append(p * num_branch[0])
        
        prompt = expanded_prompt
        batch_size = len(prompt)


        negative_prompt = negative_prompt or ""

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        else:
            assert isinstance(negative_prompt, list)

        expanded_prompt = []

        for p in negative_prompt:
            expanded_prompt.append(p*num_branch[0])
        
        negative_prompt = expanded_prompt

        # Get text embeddings
        device = self.device if not cpu_offloading else torch.device("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if use_linear_guidance:
            self._guidance_scale = guidance_scale_list[unit_index]
            self._video_guidance_scale = guidance_scale_list[unit_index]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Initialize queue and dictionary
        encoding_queue = SimpleQueue()
        latent_dict = {}

        # Setup latent dimensions and prepare initial latents
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else self.dit.config.in_channels

        # Initialize base generator for branch seeds
        base_generator = generator if generator is not None else torch.Generator(device=self.device)
        base_seed = base_generator.initial_seed() if generator is not None else torch.randint(0, 2**63-1, (1,), device=self.device).item()

        # random seeds generate
        perm_size = max(num_branch[0], 2**16)
        nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
        seeds = ((nums[:num_branch[0]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 

        # initial noisy latent list
        noisy_latents = []
        cur_height = height
        cur_width = width


        # Tree depth 0
        for i in range(num_branch[0]):
            
            seed = int(torch.randint(0, 2**31-1, (1,), device=self.device))
            branch_generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Current Encoding: {i}, seed: {seed}")

            noisy_latent, cur_height, cur_width = self.prepare_processed_latents(
                1,
                num_channels_latents,
                temp,
                height,
                width,
                prompt_embeds.dtype,
                device,
                branch_generator,
            )

            noisy_latents.append(noisy_latent)


            past_condition_latents = [[] for _ in range(len(self.stages))]
            intermed_latents = self.generate_one_unit(
                noisy_latent[:,:,:1],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                cur_height,
                cur_width,
                1,
                device,
                self.dtype,
                branch_generator,
                is_first_frame=True
            )

            # 直接存储第一帧的latents
            latent_dict[str(i)] = intermed_latents[-1]
            encoding_queue.put(str(i))

        print(f"Current Depth: 0")

        # 用于reward的编码
        reward_encodings = []
        # 是否进行reward
        reward_operate = False

        # 后续处理
        while not ( encoding_queue.empty() and not reward_operate ):

            # 若队列空,且在reward_stage中,说明要进行reward_judging了

            if reward_operate and encoding_queue.empty():
                # 使用videollama进行判断, 输出接受的视频编码(List)
                accepted_encodings = videollama_judging(output_dir, reward_encodings, prompt, reward_model, processor, self.device, reward_stages)
                reward_operate = False # 重置
                reward_encodings.clear()
                for encoding in accepted_encodings:
                    # 加入队列, 继续进行生成
                    encoding_queue.put(encoding)

            prefix_encoding = encoding_queue.get()
            current_depth = len(prefix_encoding)
            print(f"Current Depth: {current_depth}")

            if current_depth >= len(num_branch):
                history_latent_list = [] 
                # 分割prefix_encoding依次append前序的latent_dict[str]
                for i in range(current_depth):
                    partial_encoding = prefix_encoding[:i+1]
                    if partial_encoding in latent_dict:
                        history_latent_list.append(latent_dict[partial_encoding])
                
                images = self.decode_latent(torch.cat(history_latent_list,dim=2), save_memory=save_memory, inference_multigpu=inference_multigpu)
                video_path = os.path.join(output_dir, f"final_{prefix_encoding}.mp4")
                export_to_video(images, video_path, fps=24)
                print(f"Save Final Results at: {video_path}")

            else:
                if current_depth in reward_stages:
                    reward_operate = True # 可进行操作
                    
                # random seeds generate
                perm_size = max(num_branch[current_depth], 2**16)
                nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
                seeds = ((nums[:num_branch[current_depth]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 
                
                for i in range(num_branch[current_depth]):
                    
                    # Obtain history_latents again, in case of getting other frame's history
                    h_latent_list = []
                    for j in range(current_depth):
                        # 获取之前的latent作为历史信息
                        partial_encoding = prefix_encoding[:j+1]
                        if partial_encoding in latent_dict:
                            h_latent_list.append(latent_dict[partial_encoding])

                    current_encoding = prefix_encoding + str(i)
                    print(f"Current Encoding: {current_encoding}, seed: {int(seeds[i])}")
                    branch_generator = torch.Generator(device=self.device).manual_seed(int(seeds[i]))

                    # 准备条件
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(h_latent_list,dim=2), len(self.stages) - 1)
                    
                    for i_s in range(len(self.stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]
                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                        
                        # 添加多级历史帧条件
                        cur_unit_num = current_depth
                        cur_stage = i_s
                        cur_unit_ptx = 1
                        
                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    # 生成新的单元
                    intermed_latents = self.generate_one_unit(
                        noisy_latents[int(prefix_encoding[:1])][:, :, 1 + (current_depth - 1) * self.frame_per_unit:1 + current_depth * self.frame_per_unit],
                        past_condition_latents,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        cur_height,
                        cur_width,
                        self.frame_per_unit,
                        device,
                        self.dtype,
                        branch_generator,
                        is_first_frame=False
                    )

                    # 将历史latent和新生成的latent拼接在一起
                    h_latent_list.append(intermed_latents[-1])
                    combined_latents = torch.cat(h_latent_list, dim=2)  # 在时间维度上拼接
                    latent_dict[current_encoding] = intermed_latents[-1]

                    if current_depth in reward_stages:
                        # 解码完整的视频序列
                        images = self.decode_latent(combined_latents, 
                                                    save_memory=save_memory, 
                                                    inference_multigpu=inference_multigpu)
                        filename = f"path_{current_encoding}.mp4"
                        video_path = os.path.join(output_dir, filename)
                        export_to_video(images, video_path, fps=24)
                        print(f"Video saved at {os.path.join(output_dir, filename)}")
                        
                        reward_encodings.append(current_encoding)
                    
                    if not current_depth in reward_stages:
                        encoding_queue.put(current_encoding)


                    torch.cuda.empty_cache()
                    
            
        # 得到最优视频
        best_video = videollama_best_of_N(video_dir=output_dir,prompt=prompt,model=reward_model, processor=processor, device=self.device)
        print(f"Best video is {best_video}")
        
        # Clean up if using CPU offloading
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.dit.to("cpu")
            torch.cuda.empty_cache()

        return latent_dict


    

    @torch.no_grad()
    def vr_cot_gen(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]] = "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False,
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
        output_dir: Optional[str] = None,
        num_branch: List[int] = None,
        pa: bool = False,
        model = None,
        tokenizer = None,
        reward_stages: List[int] = None,
    ):

        """Vision Reward as Reward Model for CoT Video Generation.

        """
        from queue import SimpleQueue
        
        if output_dir is None:
            output_dir = "./video_cot_results"

        current_time = datetime.datetime.now()
        timestamp_dir = current_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Input validation
        if num_branch is None or len(num_branch) == 0:
            raise ValueError("num_branch must be provided and non-empty")

        # Text prompt processing
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            assert isinstance(prompt, list)

        prompt = [p + ", hyper quality, Ultra HD, 8K" for p in prompt]

        expanded_prompt = []

        for p in prompt:
            expanded_prompt.append(p * num_branch[0])
        
        prompt = expanded_prompt
        batch_size = len(prompt)


        negative_prompt = negative_prompt or ""

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        else:
            assert isinstance(negative_prompt, list)

        expanded_prompt = []

        for p in negative_prompt:
            expanded_prompt.append(p*num_branch[0])
        
        negative_prompt = expanded_prompt

        # Get text embeddings
        device = self.device if not cpu_offloading else torch.device("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if use_linear_guidance:
            self._guidance_scale = guidance_scale_list[unit_index]
            self._video_guidance_scale = guidance_scale_list[unit_index]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Initialize queue and dictionary
        encoding_queue = SimpleQueue()
        latent_dict = {}

        # Setup latent dimensions and prepare initial latents
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else self.dit.config.in_channels

        # Initialize base generator for branch seeds
        base_generator = generator if generator is not None else torch.Generator(device=self.device)
        base_seed = base_generator.initial_seed() if generator is not None else torch.randint(0, 2**63-1, (1,), device=self.device).item()

        # random seeds generate
        perm_size = max(num_branch[0], 2**16)
        nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
        seeds = ((nums[:num_branch[0]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 

        # initial noisy latent list
        noisy_latents = []
        cur_height = height
        cur_width = width


        # Tree depth 0
        for i in range(num_branch[0]):
            
            seed = int(torch.randint(0, 2**31-1, (1,), device=self.device))
            branch_generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Current Encoding: {i}, seed: {seed}")

            noisy_latent, cur_height, cur_width = self.prepare_processed_latents(
                1,
                num_channels_latents,
                temp,
                height,
                width,
                prompt_embeds.dtype,
                device,
                branch_generator,
            )

            noisy_latents.append(noisy_latent)


            past_condition_latents = [[] for _ in range(len(self.stages))]
            intermed_latents = self.generate_one_unit(
                noisy_latent[:,:,:1],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                cur_height,
                cur_width,
                1,
                device,
                self.dtype,
                branch_generator,
                is_first_frame=True
            )

            # 直接存储第一帧的latents
            latent_dict[str(i)] = intermed_latents[-1]
            encoding_queue.put(str(i))

            if pa:
                images = self.decode_latent(intermed_latents[-1], save_memory=save_memory, inference_multigpu=inference_multigpu)
                for idx, img in enumerate(images):
                    filename = f"branch_{i}_frame_0_{idx}.png"
                    img.save(os.path.join(output_dir, filename))
                    print(f"Image Saved at {os.path.join(output_dir, filename)}")

        print(f"Cuurent Depth: 0")

        reward_encodings = []
        reward_operate = False

        # 处理后续分支时
        while not ( encoding_queue.empty() and not reward_operate ):

            if reward_operate and encoding_queue.empty():
                accepted_encodings = judging(output_dir, reward_encodings, prompt, model, tokenizer, self.device, reward_stages)
                reward_operate = False # 重置
                reward_encodings.clear()
                length = len(accepted_encodings)
                depth = len(accepted_encodings[0])
                if length <= 2:
                    num_branch[depth] *= 2
                for encoding in accepted_encodings:
                    # 加入队列, 继续进行生成
                    encoding_queue.put(encoding)

                continue

            prefix_encoding = encoding_queue.get()
            current_depth = len(prefix_encoding)
            print(f"Current Depth: {current_depth}")

            if current_depth >= len(num_branch):

                history_latent_list = [] 
                # 分割prefix_encoding依次append前序的latent_dict[str]
                for i in range(current_depth):
                    partial_encoding = prefix_encoding[:i+1]
                    if partial_encoding in latent_dict:
                        history_latent_list.append(latent_dict[partial_encoding])
                
                images = self.decode_latent(torch.cat(history_latent_list,dim=2), save_memory=save_memory, inference_multigpu=inference_multigpu)
                video_path = os.path.join(output_dir, f"final_{prefix_encoding}.mp4")
                export_to_video(images, video_path, fps=24)
                print(f"Save Final Results at: {video_path}")

            else:
                if current_depth in reward_stages:
                    reward_operate = True # 可进行操作
                    
                # random seeds generate
                perm_size = max(num_branch[current_depth], 2**16)
                nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
                seeds = ((nums[:num_branch[current_depth]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 
                
                # Obtain history_latents again, in case of getting other frame's history
                h_latent_list = []
                for j in range(current_depth):
                    partial_encoding = prefix_encoding[:j+1]
                    if partial_encoding in latent_dict:
                        h_latent_list.append(latent_dict[partial_encoding])


                for i in range(num_branch[current_depth]):
                    
                    current_history = list(h_latent_list)
                    # Obtain history_latents again, in case of getting other frame's history

                    current_encoding = prefix_encoding + str(i)
                    print(f"Current Encoding: {current_encoding}, seed: {int(seeds[i])}")
                    branch_generator = torch.Generator(device=self.device).manual_seed(int(seeds[i]))

                    # 准备条件
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(current_history,dim=2), len(self.stages) - 1)
                    
                    # 从此处开始替换
                    # 输入 clean_latents_list, noisy_latents等
                    # intermed_latents = frame_wise_cot(clean_latents_list, noisy_latents, etc.)

                    for i_s in range(len(self.stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]
                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                        
                        # 添加多级历史帧条件
                        cur_unit_num = current_depth
                        cur_stage = i_s
                        cur_unit_ptx = 1
                        
                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    # 生成新的单元
                    intermed_latents = self.generate_one_unit(
                        noisy_latents[int(prefix_encoding[:1])][:, :, 1 + (current_depth - 1) * self.frame_per_unit:1 + current_depth * self.frame_per_unit],
                        past_condition_latents,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        cur_height,
                        cur_width,
                        self.frame_per_unit,
                        device,
                        self.dtype,
                        branch_generator,
                        is_first_frame=False
                    )

                    # 到此处结束替换
                    # 需要输出intermed_latents

                    # 将历史latent和新生成的latent拼接在一起
                    current_history.append(intermed_latents[-1])
                    combined_latents = torch.cat(current_history, dim=2)  # 在时间维度上拼接
                    latent_dict[current_encoding] = intermed_latents[-1]

                    if pa or current_depth in reward_stages:
                        # 解码完整的视频序列
                        images = self.decode_latent(combined_latents, 
                                                    save_memory=save_memory, 
                                                    inference_multigpu=inference_multigpu)
                        filename = f"path_{current_encoding}.mp4"
                        video_path = os.path.join(output_dir, filename)
                        export_to_video(images, video_path, fps=24)
                        print(f"Video saved at {os.path.join(output_dir, filename)}")
                        
                        if current_depth in reward_stages:
                            reward_encodings.append(current_encoding)
                    
                    if not current_depth in reward_stages:
                        encoding_queue.put(current_encoding)


                    torch.cuda.empty_cache()

        # Clean up if using CPU offloading
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.dit.to("cpu")
            torch.cuda.empty_cache()

        best_video = best_of_N(video_dir=output_dir,prompt=prompt,model=model,tokenizer=tokenizer,device=self.device)
        print(f"Best video is {best_video}")
        return best_video

    @torch.no_grad()
    def generate_img_cot(
        self,
        prompt: Union[str, List[str]] = None,
        selector: ImageSelector = None,
        file_path: str = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()


        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            batch_size = 1
            prompt = prompt + ", hyper quality, Ultra HD, 8K"        # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)
            prompt = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)

        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
                self.dit.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            # guidance_scale_list = torch.linspace(max_guidance_scale, min_guidance_scale, temp).tolist()
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        # PARM init
        clear = False
        filtered = False
        score = 1.
        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
            
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            
            image_paths = []
            latents_list = []
            images_list = []
            no_prob_list = [1.] * 10
            potential_list = [False] * 10
            for i in range(0, 10):
            
                if unit_index == 0:
                    past_condition_latents = [[] for _ in range(len(stages))]
                    intermed_latents = self.generate_one_unit(
                        latents[:,:,:1],
                        past_condition_latents,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        num_inference_steps,
                        height,
                        width,
                        1,
                        device,
                        dtype,
                        generator,
                        is_first_frame=True,
                    )
                else:
                    # prepare the condition latents
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                    
                    for i_s in range(len(stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                
                        # pad the past clean latents
                        cur_unit_num = unit_index
                        cur_stage = i_s
                        cur_unit_ptx = 1

                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                    
                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    intermed_latents = self.generate_one_unit(
                        latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                        past_condition_latents,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        height,
                        width,
                        self.frame_per_unit,
                        device,
                        dtype,
                        generator,
                        is_first_frame=False,
                    )

                latents_list.append(intermed_latents[-1])
                
                if clear: 
                    pass
                else:
                    image_filename = "waiting_for_judge.png"
                    image_path = os.path.join(file_path, image_filename)
                    generated_latents = torch.cat(generated_latents_list, dim=2)
                    images_list = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
                    for image in images_list:
                        image.save(image_path)
                        filtered, score, clear = selector.parm([prompt], image_path, clear=clear)
                
                # PARM
                frames[0].save(f"{file_path}/{i:04d}.png")
                image_paths.append(f"{file_path}/{i:04d}.png")
                no_prob_list[i] = score
                potential_list[i] = filtered
            
            true_indices = [i for i, val in enumerate(potential_list) if val]
            if true_indices:
                potential_paths = [image_paths[j] for j in true_indices]
            for i, image_path in enumerate(potential_paths):
                        
                selected, score = selector.orm([prompt], image_path)
                
                if score > highest_score_yes:
                    highest_score_yes = score
                    best_yes = i
            
            # os.system("cp " + potential_paths[best_yes] + " " + os.path.join(file_path, "best.png"))
            ret_intermed_latents = latents_list[best_yes]
            
            #generated_latents_list.append(latents_list[best_yes])
            #last_generated_latents = intermed_latents
            
        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image, filtered, score


    def generate_one_unit_img_cot(
        self,
        cur_depth, 
        latent, 
        clean_latents_list,
        prompt,
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        batch_size,
        device,
        dtype,
        generator,
        img_branch_list: List[int],
    ):
    
        score_list = [1.] * img_branch_list[cur_depth]
        latent_list = [None] * img_branch_list[cur_depth]
        highest_score_yes = float("-inf")
        for branch in range(img_branch_list[cur_depth]):
            best_yes = -1
            best_latent = None
            seed = random.randint(0, 10000)  # 随机生成一个整数作为种子
            torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
                
            if cur_depth == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents, filtered, score = self.generate_one_unit_parm(
                    latent,
                    past_condition_latents,
                    prompt,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    batch_size,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )   
            else:
                # prepare the condition latents
                past_condition_latents = []
             
                    
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                    
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents, filtered, score = self.generate_one_unit_parm(
                    latent,
                    past_condition_latents,
                    prompt,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    batch_size,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )
                    
            score_list[branch] = score    
            latent_list[branch] = intermed_latents[-1]    

            if filtered:   
                generated_latents = generated_latents_list.copy()
                generated_latents.append(intermed_latents[-1])
                generated_latents = torch.cat(generated_latents, dim=2)
                image_list = self.decode_latent(generated_latents, save_memory=True, inference_multigpu=False)
                os.makedirs(f"{file_path}/unit={unit_index}/branch={branch}", exist_ok=True)
                avg_score = 0.0
                for idx, image in enumerate(image_list[-8:]):
                    image_path = f"{file_path}/unit={unit_index}/branch={branch}/{idx:04d}.png"
                    image.save(image_path)
                    selected, score = self.selector.orm([prompt], image_path)
                    avg_score = avg_score + score
                        
                if avg_score > highest_score_yes:
                    highest_score_yes = avg_score
                    best_yes = branch
                    # best_latent = intermed_latents[-1]
            
        best_yes = min(range(len(score_list)), key=lambda i: score_list[i])        
        print("unit=",unit_index,",  best_branch=",best_yes)

        return latent_list[best_yes]


    @torch.no_grad()
    def generate_vid_cot(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()


        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        name = prompt.replace(" ", "_")
        file_path = f"outputs/{name}"
        
        prompt_list = split_prompt(prompt)
        print(prompt_list)
        if isinstance(prompt, str):
            batch_size = 1
            prompt = prompt + ", hyper quality, Ultra HD, 8K"        # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)
            prompt = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)

        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
                self.dit.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            # guidance_scale_list = torch.linspace(max_guidance_scale, min_guidance_scale, temp).tolist()
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        
        # PARM init
        score = 1.
        self.selector = ImageSelector(device=self.device, device_map={"": device}, pretrained='/home/lff/data1/why/videocot/Image-Generation-CoT/ckpts/parm')

        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
            
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]
            
            # 使用不同的prompt
            if unit_index == 0:
                prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt_list[0], device)
            elif unit_index < temp-1:
                prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt_list[1], device)
            else:
                prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt_list[2], device)

            # cfg
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            
            os.makedirs(f"{file_path}/unit={unit_index}", exist_ok=True)
            
            # branch_num = 5
            branch_num_list = [20, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
            
            # Replace
            intermed_latents = generate_one_unit_img_cot(cur_depth, noisy_latents, ... ,clean_latents_list,)


            # Start of function
            score_list = [1.] * branch_num_list[unit_index]
            latent_list = [None] * branch_num_list[unit_index]
            highest_score_yes = float("-inf")
            for branch in range(branch_num_list[unit_index]):
                best_yes = -1
                best_latent = None
                seed = random.randint(0, 10000)  # 随机生成一个整数作为种子
                torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
                
                if unit_index == 0:
                    past_condition_latents = [[] for _ in range(len(stages))]
                    intermed_latents, filtered, score = self.generate_one_unit(
                        latents[:,:,:1],
                        past_condition_latents,
                        prompt,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        num_inference_steps,
                        height,
                        width,
                        1,
                        device,
                        dtype,
                        generator,
                        is_first_frame=True,
                    )   
                else:
                    # prepare the condition latents
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                    
                    for i_s in range(len(stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                
                        # pad the past clean latents
                        cur_unit_num = unit_index
                        cur_stage = i_s
                        cur_unit_ptx = 1

                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                    
                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    intermed_latents, filtered, score = self.generate_one_unit(
                        latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                        past_condition_latents,
                        prompt,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        height,
                        width,
                        self.frame_per_unit,
                        device,
                        dtype,
                        generator,
                        is_first_frame=False,
                    )
                    
                score_list[branch] = score    
                latent_list[branch] = intermed_latents[-1]    
                if filtered:   
                    generated_latents = generated_latents_list.copy()
                    generated_latents.append(intermed_latents[-1])
                    generated_latents = torch.cat(generated_latents, dim=2)
                    image_list = self.decode_latent(generated_latents, save_memory=True, inference_multigpu=False)
                    os.makedirs(f"{file_path}/unit={unit_index}/branch={branch}", exist_ok=True)
                    avg_score = 0.0
                    for idx, image in enumerate(image_list[-8:]):
                        image_path = f"{file_path}/unit={unit_index}/branch={branch}/{idx:04d}.png"
                        image.save(image_path)
                        selected, score = self.selector.orm([prompt], image_path)
                        avg_score = avg_score + score
                        
                    if avg_score > highest_score_yes:
                        highest_score_yes = avg_score
                        best_yes = branch
                        # best_latent = intermed_latents[-1]
            
            best_yes = min(range(len(score_list)), key=lambda i: score_list[i])        
            # generated_latents_list.append(intermed_latents[-1])
            print("unit=",unit_index,",  best_branch=",best_yes)
            generated_latents_list.append(latent_list[best_yes])

            # end of generated latents
            last_generated_latents = intermed_latents
            
        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image        


    @torch.no_grad()
    def video_cot_gen(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]] = "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False,
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
        output_dir: Optional[str] = None,
        video_branch_list: List[int] = None, 
        img_branch_list: List[int] = None, 
        pa: bool = False,
        model = None,
        processor = None,
        tokenizer = None,
        mode:str = "vl",
        reward_stages: List[int] = None,
    ):
        """Chain-of-Thought Video Generation

            video_branch_list: Number of branches for each depth in temporal video generation.
            img_branch_list: Number of branches for each depth in spatial image generation.
            model: vlm-based video judgement model. (VideoLLaMA)
            processor: processor (VideoLLaMA's processor)
            tokenizer: tokenizer (VisionReward's tokenizer)
            mode: use which vlm as reward model to judge and select. 'vr' = 'visionreward', 'vl'='videollama'
            reward_stages: stage(depth) for reward model working.
        
        Output: 
            Videos, as the generation result.
        """

        from queue import SimpleQueue
        
        # 创建输出文件夹
        if output_dir is None:
            output_dir = "./video_tree_generate"
        current_time = datetime.datetime.now()
        timestamp_dir = current_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Input validation
        if img_branch_list is None or len(img_branch_list) == 0:
            raise ValueError("num_branch must be provided and non-empty")
        
        name = prompt.replace(" ", "_")
        file_path = f"outputs/{name}"
        

        # prepare hierarchical prompts
        initial_prompt = prompt
        prompt_list = split_prompt(prompt)
        print(prompt_list)

        if isinstance(prompt_list, str):
            batch_size = 1
            prompt_list = prompt_list + ", hyper quality, Ultra HD, 8K"        # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt_list, list)
            batch_size = len(prompt_list)
            prompt_list = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt_list]


        negative_prompt = negative_prompt or ""

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        else:
            assert isinstance(negative_prompt, list)

        # Get text embeddings
        device = self.device if not cpu_offloading else torch.device("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if use_linear_guidance:
            self._guidance_scale = guidance_scale_list[unit_index]
            self._video_guidance_scale = guidance_scale_list[unit_index]

        # Initialize queue and dictionary
        encoding_queue = SimpleQueue()
        latent_dict = {}

        # Setup latent dimensions and prepare initial latents
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else self.dit.config.in_channels

        # Initialize base generator for branch seeds
        base_generator = generator if generator is not None else torch.Generator(device=self.device)
        base_seed = base_generator.initial_seed() if generator is not None else torch.randint(0, 2**63-1, (1,), device=self.device).item()

        # random seeds generate
        perm_size = max(num_branch[0], 2**16)
        nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
        seeds = ((nums[:num_branch[0]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 

        # initial noisy latent list
        noisy_latents = []
        cur_height = height
        cur_width = width
        
        # Tree depth 0
        for i in range(num_branch[0]):

            # Use Prompt for Stage 1 (T2I)
            prompt = prompt_list[0]
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            
            seed = int(torch.randint(0, 2**31-1, (1,), device=self.device))
            branch_generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Current Encoding: {i}, seed: {seed}")

            noisy_latent, cur_height, cur_width = self.prepare_processed_latents(
                1,
                num_channels_latents,
                temp,
                height,
                width,
                prompt_embeds.dtype,
                device,
                branch_generator,
            )

            noisy_latents.append(noisy_latent)

            past_condition_latents = [[] for _ in range(len(self.stages))]

            intermed_latents = self.generate_one_unit_img_cot(
                0,
                noisy_latent[:,:,:1],
                None,
                prompt,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                cur_height,
                cur_width,
                1,
                device,
                self.dtype,
                branch_generator,
                img_branch_list,
            )

            # 直接存储第一帧的latents
            latent_dict[str(i)] = intermed_latents[-1]
            encoding_queue.put(str(i))

        print(f"Current Depth: 0")

        # 用于reward的编码
        reward_encodings = []
        # 是否进行reward
        reward_operate = False
        # prompt改变
        prompt_change = True
        # prompt
        prompt = None

        # 后续处理
        while not ( encoding_queue.empty() and not reward_operate ):
            
            # 若队列空,且在reward_stage中,说明要进行reward_judging了
            if reward_operate and encoding_queue.empty():
                # 使用videollama进行判断, 输出接受的视频编码(List)
                if mode == 'vl':
                    accepted_encodings = videollama_judging(output_dir, reward_encodings, prompt, model, processor, self.device, reward_stages)
                elif mode == 'vr':
                    accepted_encodings = judging(output_dir, reward_encodings, prompt, model, tokenizer, self.device, reward_stages)
                else:
                    print(f"Error: reward mode is not set correctly(vr or vl).")
                reward_operate = False # 重置
                reward_encodings.clear()
                for encoding in accepted_encodings:
                    # 加入队列, 继续进行生成
                    encoding_queue.put(encoding)
            
            # 其余情况,确定目前的编码,并进行text prompt选择以及处理
            prefix_encoding = encoding_queue.get()
            current_depth = len(prefix_encoding)
            print(f"Current Depth: {current_depth}")

            # 更改text prompt
            if prompt_change:
                if current_depth < temp - 1:
                    prompt_change = False
                    prompt = prompt_list[1]
                    prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
                    if self.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                
                else:
                    prompt_change = False
                    prompt = prompt_list[2]
                    prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
                    if self.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

            # 输出
            if current_depth >= len(num_branch):
                history_latent_list = [] 
                # 分割prefix_encoding依次append前序的latent_dict[str]
                for i in range(current_depth):
                    partial_encoding = prefix_encoding[:i+1]
                    if partial_encoding in latent_dict:
                        history_latent_list.append(latent_dict[partial_encoding])
                
                images = self.decode_latent(torch.cat(history_latent_list,dim=2), save_memory=save_memory, inference_multigpu=inference_multigpu)
                video_path = os.path.join(output_dir, f"final_{prefix_encoding}.mp4")
                export_to_video(images, video_path, fps=24)
                print(f"Save Final Results at: {video_path}")

            else:
                if current_depth in reward_stages:
                    reward_operate = True # 可进行操作
                
                # random seeds generate
                perm_size = max(num_branch[current_depth], 2**16)
                nums = torch.randperm(perm_size, generator=base_generator, device=self.device)
                seeds = ((nums[:num_branch[current_depth]].to(torch.int64) * 2**32) + base_seed) % (2**32 - 1) 
                
                for i in range(num_branch[current_depth]):
                    
                    # Obtain history_latents again, in case of getting other frame's history
                    h_latent_list = []
                    for j in range(current_depth):
                        # 获取之前的latent作为历史信息
                        partial_encoding = prefix_encoding[:j+1]
                        if partial_encoding in latent_dict:
                            h_latent_list.append(latent_dict[partial_encoding])

                    current_encoding = prefix_encoding + str(i)
                    print(f"Current Encoding: {current_encoding}, seed: {int(seeds[i])}")
                    branch_generator = torch.Generator(device=self.device).manual_seed(int(seeds[i]))

                    # 准备条件
                    past_condition_latents = []
                    clean_latents_list = self.get_pyramid_latent(torch.cat(h_latent_list,dim=2), len(self.stages) - 1)
                    
                    for i_s in range(len(self.stages)):
                        last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]
                        stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                        
                        # 添加多级历史帧条件
                        cur_unit_num = current_depth
                        cur_stage = i_s
                        cur_unit_ptx = 1
                        
                        while cur_unit_ptx < cur_unit_num:
                            cur_stage = max(cur_stage - 1, 0)
                            if cur_stage == 0:
                                break
                            cur_unit_ptx += 1
                            cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                            cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                            stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                        stage_input = list(reversed(stage_input))
                        past_condition_latents.append(stage_input)

                    # 生成新的单元
                    intermed_latents = self.generate_one_unit_img_cot(
                        cur_depth,
                        noisy_latents[int(prefix_encoding[:1])][:, :, 1 + (current_depth - 1) * self.frame_per_unit:1 + current_depth * self.frame_per_unit],
                        clean_latents_list,
                        prompt,
                        prompt_embeds,
                        prompt_attention_mask,
                        pooled_prompt_embeds,
                        video_num_inference_steps,
                        cur_height,
                        cur_width,
                        1,
                        device,
                        self.dtype,
                        branch_generator,
                        img_branch_list,
                    )

                    # 将历史latent和新生成的latent拼接在一起
                    h_latent_list.append(intermed_latents[-1])
                    combined_latents = torch.cat(h_latent_list, dim=2)  # 在时间维度上拼接
                    latent_dict[current_encoding] = intermed_latents[-1]

                    if current_depth in reward_stages:
                        # 解码完整的视频序列
                        images = self.decode_latent(combined_latents, 
                                                    save_memory=save_memory, 
                                                    inference_multigpu=inference_multigpu)
                        filename = f"path_{current_encoding}.mp4"
                        video_path = os.path.join(output_dir, filename)
                        export_to_video(images, video_path, fps=24)
                        print(f"Video saved at {os.path.join(output_dir, filename)}")
                        
                        reward_encodings.append(current_encoding)
                    
                    if not current_depth in reward_stages:
                        encoding_queue.put(current_encoding)


                    torch.cuda.empty_cache()
                    
            
        # 得到最优视频
        if mode == 'vl':
            best_video = videollama_best_of_N(video_dir=output_dir,prompt=initial_prompt,model=model, processor=processor, device=self.device)
        elif mode == 'vr':
            best_video = best_of_N(output_dir, initial_prompt, model, tokenizer, device=self.device)           
        else:
            print(f"Error: reward mode is not set correctly (vr or vl).")
        
        print(f"Best video is {best_video}")
        
        # Clean up if using CPU offloading
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.dit.to("cpu")
            torch.cuda.empty_cache()

        return best_video