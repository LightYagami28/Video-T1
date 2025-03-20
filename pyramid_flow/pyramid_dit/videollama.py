import torch
import os
from tqdm import tqdm
from typing import List, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
import io
from decord import cpu, VideoReader, bridge
import json

def load_video(video_data, strategy='base', max_frames=180, fps=1):
    """Load and preprocess video data.
    
    Args:
        video_data (bytes): Raw video data
        strategy (str): Strategy for frame selection ('base' or 'chat')
        max_frames (int): Maximum number of frames to extract
        fps (int): Frames per second to extract
        
    Returns:
        torch.Tensor: Processed video frames
    """
    bridge.set_bridge('torch')
    decord_vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    total_frames = len(decord_vr)
    
    if strategy == 'base':
        # Extract frames at specified FPS up to max_frames
        frame_indices = list(range(0, total_frames, int(decord_vr.get_avg_fps() / fps)))
        if len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]
    else:  # strategy == 'chat'
        # Extract frames based on timestamps
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [t[0] for t in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_indices = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_indices.append(index)
            if len(frame_indices) >= max_frames:
                break
    
    video_frames = decord_vr.get_batch(frame_indices)
    # Convert to NCHW format
    video_frames = video_frames.permute(3, 0, 1, 2)
    return video_frames

def videollama_best_of_N(
    video_dir: str,
    prompt: Union[str, List[str]],
    model,
    processor,
    device: str = "cuda:0",
    question_path: str = "./VisionReward_video_qa_select.txt",
    weight_path: str = "./weight.json"
) -> str:
    """Select the best video from multiple candidates using VideoLLaMA model.
    
    Args:
        video_dir (str): Directory containing videos to evaluate
        prompt (Union[str, List[str]]): Prompt used to generate the videos
        model_path (str): Path to the VideoLLaMA model
        device (str): Device to run the model on
        question_path (str): Path to questions file
        weight_path (str): Path to weights file
    
    Returns:
        str: Filename of the best video
    """
    # Load model and processor
    # 如果没有传Model/processor就加载，但是一般video_cot要预先load model，不然每次加载太费时
    model_path = "../VideoLLaMA3/VideoLLaMA3-7B"
    if not model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    
    if not processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load questions and weights
    with open(question_path, 'r') as f:
        questions = f.readlines()
    with open(weight_path, 'r') as f:
        weights = json.load(f)

    if isinstance(prompt, list):
        prompt = ' '.join(prompt)

    def score_video(video_path: str, prompt: str) -> float:
        """Calculate score for a single video."""
        queries = [question.replace('[[prompt]]', prompt) for question in questions]
        answers = []
        
        
        for query in tqdm(queries, 'scoring video'):
            # 这里video_data给的是path,而不是load后的data,fps是每秒加载多少张图片
            conversation = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question only use yes or no, no other descriptions."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_path, "fps":4, "max_frames": 180}},
                        {"type": "text", "text": query},
                    ]
                },
            ]
            
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=1024)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                answers.append(response)
        
        # Convert answers to scores (-1 for 'no', 1 for 'yes')
        answer_scores = np.array([1 if answer.lower() == 'yes' else -1 for answer in answers])
        return np.mean(answer_scores * weights).item()

    # Find all videos with format final_*.mp4
    video_files = [f for f in os.listdir(video_dir) if f.startswith('final_') and f.endswith('.mp4')]
    
    best_score = float('-inf')
    best_video = None

    # Score each video and find the best one
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        score = score_video(video_path, prompt)
        if score > best_score:
            best_score = score
            best_video = video_file

    # Save results
    result_path = os.path.join(video_dir, "best_of_N_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Best Video: {best_video}, Score: {best_score}")

    return best_video

def videollama_judging(
    output_dir: str,
    encodings: List,
    prompt: Union[str, List[str]],
    model,
    processor,
    device: str = "cuda:0",
    reward_stages: List[int] = None,
    judge_mode: str = "pct",
    question_paths: dict = {
        0: "./question0.txt",
        1: "./question1.txt",
        2: "./question2.txt"
    }
) -> List:
    """Judge generated videos using VideoLLaMA model.
    
    Args:
        output_dir (str): Directory containing output videos
        encodings (List): Videos to be judged
        prompt (Union[str, List[str]]): Prompt used to generate videos
        model_path (str): Path to VideoLLaMA model
        device (str): Device to run the model on
        reward_stages (List[int]): Total reward stages
        judge_mode (str): Mode for judging ('pct' or 'pf')
        question_paths (dict): Paths to question files for each stage
        
    Returns:
        List: Accepted video encodings
    """
    # Load model and processor
    model_path = "../VideoLLaMA3/VideoLLaMA3-7B"
    if not model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    if not processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if isinstance(prompt, list):
        prompt = ' '.join(prompt)

    # Determine current stage
    current_depth = len(encodings[0]) - 1
    stage = reward_stages.index(current_depth)

    # Load questions for current stage
    with open(question_paths[stage], 'r', encoding='utf-8') as f:
        questions = f.readlines()
    
    queries = [question.replace('[[prompt]]', prompt).strip() for question in questions]
    results = []

    # Evaluate each video
    for encoding in encodings:
        video_path = os.path.join(output_dir, f"path_{encoding}.mp4")
        
        answers = []
        # conversation同上
        for query in tqdm(queries, desc=f"scoring video {encoding}"):
            conversation = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question only use yes or no, no other descriptions."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_path, "fps":4, "max_frames": 180}},
                        {"type": "text", "text": query},
                    ]
                },
            ]
            
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=1024)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                answers.append(response)

        # Calculate scores
        array_answer = np.array([1 if answer.lower() == "yes" else -1 for answer in answers])
        score = int(array_answer.sum())
        all_yes = np.all(array_answer == 1)

        results.append({
            "encoding": encoding,
            "score": score,
            "answers": answers,
            "all_yes": all_yes,
        })

    # Process results based on judge_mode
    accepted_encodings = []
    half_accepted = []

    if judge_mode == "pf":
        for res in results:
            if res["all_yes"]:
                accepted_encodings.append(res["encoding"])
            elif res["score"] > 0:
                half_accepted.append(res["encoding"])
        
        if accepted_encodings:
            output = accepted_encodings
        elif half_accepted:
            output = half_accepted
        else:
            output = [max(results, key=lambda x: x["score"])["encoding"]]
    
    elif judge_mode == "pct":
        scores = [res["score"] for res in results]
        median_score = np.median(scores)
        accepted_encodings = [res["encoding"] for res in results if res["score"] >= median_score]
        output = accepted_encodings if accepted_encodings else [max(results, key=lambda x: x["score"])["encoding"]]
    
    else:
        raise ValueError(f"Unsupported judge_mode: {judge_mode}")

    # Save results
    output_filename = os.path.join(output_dir, f"Reward_{stage}.txt")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Accepted Encodings: {output}")

    return output