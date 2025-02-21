import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7080/v1", api_key="qwen2.5-o1")

system_message = [{"role": "system", "content": "你是一个数据分类专家，请根据对话内容判断其所属的类别。"}]
last_query = [{"role": "user", "content": "现在请输出你的判断结果："}]

def infer_data_cls(messages):
    response = client.chat.completions.create(
        model="QWen2.5-O1",
        messages=messages
    )
    return response.choices[0].message.content

def process_item(item):
    # 构造对话信息
    conversations = item.get("conversations", [])
    message = [{"role": m["from"], "content": m["value"]} for m in conversations]
    messages = system_message + message + last_query
    task_type = infer_data_cls(messages)
    item["task_type"] = task_type
    return item

def infer_data_cls_from_file(input_path, output_path, max_workers=10):
    # 读取数据文件
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 仅处理数据的部分内容，假设从第100000项开始
    data_slice = data[100000:]
    processed_results = []
    # 使用线程池并行处理数据
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in data_slice}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result_item = future.result()
                processed_results.append(result_item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")

    # 写入处理结果到文件
    with open(output_path, "w") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)
    logger.info(f"Processed {len(processed_results)} data cls done, save to {output_path}")

if __name__ == "__main__":
    input_path = "/root/app/rag_data/qwen_bench/qwen_bench_300k.json"
    output_path = "/root/app/rag_data/qwen_bench/data_clsed/qwen_bench_200k_cls.json"
    # 根据环境和 API 限制，调整 max_workers 的数量
    infer_data_cls_from_file(input_path, output_path, max_workers=10)
