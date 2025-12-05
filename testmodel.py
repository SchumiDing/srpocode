import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



model_name = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/tmp"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map=device, 
    # trust_remote_code=True,
).eval()


prompt = """\n\nFind the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.\nLet's think step by step and put the final answer in the \\boxed{} tag. Do not repeat any sentences in the answer, and keep only one \\boxed{} tag which contains the final answer."""
ans = "34"

# tokenizer.chat_template = """
#     {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

# '+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

# ' }}{% endif %}
#     """

messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(text)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=1
)  # (B, L')

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# prompt2 = """\n Given a right triangle with two legs of length 3 and 4, find the length of the hypotenuse.

# Let's think step by step and put the final answer in the \\boxed{} tag. Do not repeat any sentences in the answer, and keep only one \\boxed{} tag which contains the final answer."""

# messages2 = [{"role": "user", "content": prompt2}]

# text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)

# print(text2)

# inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)

# outputs2 = model.generate(**inputs2, max_new_tokens=2048)

# print(tokenizer.decode(outputs2[0], skip_special_tokens=True))

