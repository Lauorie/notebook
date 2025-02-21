import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from loguru import logger

model_name = "/root/app/ms-swift/output/qwen2_5-7b/数据分类/checkpoint-362"

# 加载模型和 tokenizer，同时调整 padding_side 为 left（适用于 decoder-only 模型）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# 对话模板
system_message = [{"role": "system", "content": "你是一个数据分类专家，请根据对话内容判断其所属的类别。"}]
last_query = [{"role": "user", "content": "现在请输出你的判断结果："}]

def prepare_text(messages: list[dict]) -> str:
    """
    将 messages 中的 "from"/"value" 键转为 "role"/"content"，并构造完整对话文本
    """
    messages = [{"role": item["from"], "content": item["value"]} for item in messages]
    messages = system_message + messages + last_query
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def generate_task_types_batch(messages_batch: list[list[dict]]) -> list[str]:
    """
    对一个 batch 的对话列表进行推理生成，并返回每个对话中 assistant 的回答部分
    """
    # 将每个消息列表转换为完整文本
    texts = [prepare_text(messages) for messages in messages_batch]
    
    # 使用批量编码，并进行 padding 以适应批量输入
    model_inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            eos_token_id=[151643, 151645],
            pad_token_id=151643,
            do_sample=True,
            repetition_penalty=1.05, 
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    task_types = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return task_types

def process_json(json_path: str, save_path: str, batch_size: int = 8):
    """
    读取 JSON 文件，对从第 100k 条以后的数据进行批量推理处理，
    并将结果写回保存。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 只处理 10w 条以后的数据
    data_slice = data[100000:]
    
    # 分批处理，batch_size 可根据 GPU 显存情况进行调整
    for i in tqdm(range(0, len(data_slice), batch_size)):
        batch = data_slice[i : i + batch_size]
        conversations_batch = [item["conversations"] for item in batch]
        task_types = generate_task_types_batch(conversations_batch)
        for item, answer in zip(batch, task_types):
            item["task_type"] = answer

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_slice, f, ensure_ascii=False, indent=4)
    logger.info(f"已处理 {len(data_slice)} 条数据，保存到 {save_path}")
if __name__ == "__main__":
    json_path = "/root/app/rag_data/qwen_bench/qwen_bench_300k.json"
    save_path = "/root/app/rag_data/qwen_bench/data_clsed/qwen_bench_200k_cls.json"
    process_json(json_path, save_path, batch_size=16)
