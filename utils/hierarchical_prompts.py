import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def hierarchical_prompts(prompt, device, model_path=None):
    """
    Using a LLM to generate Hierarchical Prompts.
        Input: a general prompt
        Output: a prompt list with 3 stages [..., ..., ...]
    """
    # Load model and tokenizer
    if not model_path:
        model_path = "/mnt/public/huggingface/DeepSeek-R1-Distill-Llama-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map={"": device}, torch_dtype="auto")

    # System prompt
    system_prompt = """Split video generation prompts into 3 stages:
1. Static scene description. Describe all the objects or characters in the input prompt.
2. Action/motion directions. Describe the motion or action of all the objects or characters in the input prompt.
3. Expected ending state. Describe the ending scene of the video.
Here is an input-output example:
Input: a girl pushing the chair while her sister is on the chair
Output: [
    "A girl standing behind a chair with her sister sitting on it in a living room, realistic lighting",
    "Smooth pushing motion showing chair movement, sister holding armrests, hair movement",
    "Chair positioned several feet away from original spot, sister smiling while sitting securely"
]
You need to give output in a list ["","",""]
Now process this input:"""

    # Generate answer
    inputs = tokenizer(f"{system_prompt} {prompt}. Remember that you must follow the example format and output only a list.", return_tensors="pt").to(device)
    while True:
        outputs = model.generate(**inputs, max_length=5120)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract result list
        start_idx = result.rfind("[")
        end_idx = result.rfind("]") + 1
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            continue
        try:
            prompt_list = eval(result[start_idx:end_idx])
            # Validate the list
            if (isinstance(prompt_list, list) and len(prompt_list) == 3 and all(isinstance(item, str) for item in prompt_list) and prompt_list[0] != "A girl standing behind a chair with her sister sitting on it in a living room, realistic lighting"):
                return prompt_list
        except Exception:
            # If eval fails due to syntax errors, retry
            pass

    return eval(result[start_idx:end_idx])
