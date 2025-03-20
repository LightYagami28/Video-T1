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
        # print(result)
        start_idx = result.rfind("[")
        end_idx = result.rfind("]") + 1
        prompt_list = eval(result[start_idx:end_idx])
        
        # Check conditions
        if start_idx < end_idx and prompt_list[0] != "A girl standing behind a chair with her sister sitting on it in a living room, realistic lightin":
            break

    return eval(result[start_idx:end_idx])
