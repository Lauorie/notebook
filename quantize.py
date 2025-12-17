
# pip install torch
# pip install transformers==4.51.3 autoawq-kernels autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/root/app/shared/Qwen3-32B-FT/v1-20251209-134423/checkpoint-2000"
quant_path = "/root/app/shared/Qwen3-32B-FT/v1-20251209-134423/checkpoint-2000-AWQ"


# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto", safetensors=True
)



# The pattern of data
""" # Example
msg=[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]
data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())
return data
"""


def data_gen():
    import json

    data = []
    with open("/root/app/eval_verify_not_cot_350.json", "r", encoding="utf-8") as file:
        cot_data = json.load(file)[:64]
    for item in cot_data:
        msg = item["messages"]
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=False
        )
        data.append(text.strip())
    return data



# Quantize
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=data_gen(),
    n_parallel_calib_samples=1,
    max_calib_samples=256,
    max_calib_seq_len=1024,
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
