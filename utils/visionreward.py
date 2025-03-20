import torch
import os
import shutil
import re
import numpy as np
import io
import json
from tqdm import tqdm

def best_of_N(
    video_dir,
    result_dir,
    prompt,
    model,
    tokenizer,
    device,
    video_name=None
):
    from decord import cpu, VideoReader, bridge
    """Using Vision Reward Model as Reward, get score of videos and select best of N.
    """

    QUESTION_PATH = "./vqa/VisionReward_video_qa_select.txt"
    WEIGHT_PATH = "./vqa/weight.json"

    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    with open(QUESTION_PATH, 'r') as f:
        questions = f.readlines()

    with open(WEIGHT_PATH, 'r') as f:
        weight = json.load(f)

    if isinstance(prompt, list):
        prompt = ' '.join(prompt)

    def load_video(video_data, strategy='chat'):
        bridge.set_bridge('torch')
        mp4_stream = video_data
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames,
                            int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

    def inference(video_path, query, temperature=0.1):
        video_data = open(video_path, 'rb').read()              
        strategy = 'base'
        video = load_video(video_data, strategy=strategy)
        
        history = []

        inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=query,
                images=[video],
                history=history,
                template_version=strategy
            )

        inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
            }
        gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,
                "do_sample": False,
            }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]]
        return tokenizer.decode(outputs[0])

    def score(video_path, prompt) -> float:
        queries = [question.replace('[[prompt]]', prompt) for question in questions]
        answers = []
        for query in tqdm(queries, 'scoring video'):
            answer = inference(video_path, query)
            answers.append(answer)
        answers = np.array([1 if answer == 'yes' else -1 for answer in answers])
        return np.mean(answers * weight).item()
    


    # find all videos in the format: final_*.mp4
    video_files = [f for f in os.listdir(video_dir) if f.startswith('final_') and f.endswith('.mp4')]
    
    best_score = -float('inf')
    best_video = None

    # for each video, calculate their scores.
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        s = score(video_path, prompt)
        # use the score to choose best among them.
        if s > best_score:
            best_score = s
            best_video = video_file

    print(f"Best Video: {best_video}")

    filename = os.path.join(video_dir, "best_of_N_result.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Best Video: {best_video}, Score:{best_score}")

    # Create a valid filename from the prompt
    def create_valid_filename(s):
        s = s.replace(' ', '_')
        # Replace invalid characters with underscore
        s = re.sub(r'[<>:"/\\|?*]', '_', s)
        # Limit length to 255 characters
        s = s[:255]
        # Remove leading/trailing spaces and periods
        s = s.strip('. ')
        return s if s else 'untitled'

    if best_video:
        # Create new filename based on prompt
        if not video_name: 
            new_filename = create_valid_filename(prompt) + '.mp4'
        else:
            new_filename = video_name + '.mp4'
            
        # Copy the best video to result_dir with new name
        src_path = os.path.join(video_dir, best_video)
        dst_path = os.path.join(result_dir, new_filename)
        shutil.copy2(src_path, dst_path)
        print(f"Best video saved as: {dst_path}")

    return best_video

    

def judging(
    video_dir,
    encodings,
    prompt,
    model,
    tokenizer,
    device,
    reward_stages,
    judge_mode: str = "pct",
):
    """Using Vision Reward Model as Reward, judge generated videos in different perspectives.

    Args:
        output_dir (str): Directory saving output videos.
        encodings (List): Videos to be judged.
        prompt (str): Prompt used to generate video.
        model: VLM-based model used as reward.
        tokenizer: Reward model's tokenizer.
        device: Device identifier.
        reward_stages: Total reward stages.
        judge_mode: Mode for judging
            - pct: For videos pass or in the top 50% score, go to next stage
            - pf: pass/half/highest score.

    Returns:
        List: Represent videos accepted by the reward model.
    """
    from decord import cpu, VideoReader, bridge

    QUESTION_0_PATH = "./vqa/question0.txt"
    QUESTION_1_PATH = "./vqa/question1.txt"
    QUESTION_2_PATH = "./vqa/question2.txt"
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def load_video(video_data, strategy="chat"):
        bridge.set_bridge("torch")
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
        total_frames = len(decord_vr)
        frame_id_list = []

        if strategy == "base":
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == "chat":
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            # decord returns a tuple for each timestamp, so extract the first element.
            timestamps = [t[0] for t in timestamps]
            max_second = round(max(timestamps)) + 1
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break

        video_frames = decord_vr.get_batch(frame_id_list)
        video_frames = video_frames.permute(3, 0, 1, 2)
        return video_frames

    results = []

    # Identify the stage based on current depth.
    current_depth = len(encodings[0]) - 1
    stage = reward_stages.index(current_depth)
    
    if stage == 0:
        with open(QUESTION_0_PATH, "r", encoding="utf-8") as f:
            questions = f.readlines()
    elif stage == 1:
        with open(QUESTION_1_PATH, "r", encoding="utf-8") as f:
            questions = f.readlines()
    else:
        with open(QUESTION_2_PATH, "r", encoding="utf-8") as f:
            questions = f.readlines()

    strategy = "base"

    # Ensure prompt is a string.
    if isinstance(prompt, list):
        prompt = " ".join(prompt)

    # Replace placeholder with the prompt in each question and strip extra whitespace.
    queries = [question.replace("[[prompt]]", prompt).strip() for question in questions]
    
    for encoding in encodings:
        path = os.path.join(video_dir, f"path_{encoding}.mp4")
        with open(path, "rb") as f:
            video_data = f.read()
        video = load_video(video_data, strategy=strategy)

        history = []
        answers = []
        for query in tqdm(queries, desc=f"scoring video {encoding}"):
            inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=query,
                images=[video],
                history=history,
                template_version=strategy,
            )

            # Move inputs to the proper device and tensor type.
            inputs = {
                "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
                "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
                "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
                "images": [[inputs["images"][0].to(device).to(TORCH_TYPE)]],
            }
            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,
                "do_sample": False,
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                # Assuming that new tokens are appended after the input_ids.
                outputs = outputs[:, inputs["input_ids"].shape[1]:]
            decoded_output = tokenizer.decode(outputs[0])
            cleaned_output = decoded_output.replace("<|end_of_text|>", "").strip()
            answers.append(cleaned_output)
            history.append((query, cleaned_output))

        # Map "yes" to 1 and any other answer to -1 (case insensitive).
        array_answer = np.array([1 if answer.lower() == "yes" else -1 for answer in answers])
        score = int(array_answer.sum())
        all_yes = np.all(array_answer == 1)

        results.append({
            "encoding": encoding,
            "score": score,
            "answers": answers,
            "all_yes": all_yes,
        })

    accepted_encodings = []
    half_accepted = []

    # Decide based on judge_mode.
    if judge_mode == "pf":
        # Pass if all answers are "yes", half-accepted if score > 0.
        for res in results:
            if res["all_yes"]:
                accepted_encodings.append(res["encoding"])
            elif res["score"] > 0:
                half_accepted.append(res["encoding"])
        if accepted_encodings:
            output_filename = os.path.join(video_dir, f"Reward_{stage}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"Accepted Encodings: {accepted_encodings}")
            return accepted_encodings
        elif half_accepted:
            output_filename = os.path.join(video_dir, f"Reward_{stage}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"No accepted encoding, half-correct: {half_accepted}")
            return half_accepted
        else:
            # In case no video meets the above criteria, return the video with the highest score.
            best_encoding = max(results, key=lambda x: x["score"])["encoding"]
            output_filename = os.path.join(video_dir, f"Reward_{stage}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"No video accepted, return relative good one: {best_encoding}")
            return [best_encoding]
    elif judge_mode == "pct":
        # Accept videos with score at or above the median.
        scores_list = [res["score"] for res in results]
        median_score = np.median(scores_list) if scores_list else 0
        for res in results:
            if res["score"] >= median_score:
                accepted_encodings.append(res["encoding"])
        if accepted_encodings:
            output_filename = os.path.join(video_dir, f"Reward_{stage}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"Accepted Encodings (score >= median {median_score}): {accepted_encodings}")
            return accepted_encodings
        else:
            # Fallback to best encoding.
            best_encoding = max(results, key=lambda x: x["score"])["encoding"]
            output_filename = os.path.join(video_dir, f"Reward_{stage}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"No video accepted, return relative good one: {best_encoding}")
            return [best_encoding]
    else:
        raise ValueError(f"Unsupported judge_mode: {judge_mode}")
