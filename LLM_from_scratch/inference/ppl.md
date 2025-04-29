# 对话数据困惑度 (Perplexity) 计算脚本

脚本 (`ppl.py`) 用于计算给定对话数据集中每个样本的困惑度 (Perplexity, PPL)。它主要用于评估语言模型在生成特定响应时的流畅性和合理性。脚本会读取一个包含对话数据的 JSON 文件，使用指定的 Hugging Face Causal LM 模型计算每个对话中助手响应部分的 PPL，并将带有 PPL 分数的结果保存到新的 JSON 文件中。

## 功能

*   加载 Hugging Face 的 AutoTokenizer 和 AutoModelForCausalLM。
*   读取指定格式的 JSON 对话数据。
*   对数据进行批处理以提高效率。
*   计算每个对话中 **助手 (assistant)** 回答部分的 Perplexity。
*   将计算出的 PPL 值添加到原始数据中，并保存为新的 JSON 文件。
*   使用 `torch.amp.autocast` 进行混合精度推理以加速计算并节省显存。
*   处理空的或无效的助手响应。

## 依赖

确保已安装所需的 Python 库：

```bash
pip install torch transformers tqdm
```

## 配置

在运行脚本之前，需要修改 `ppl.py` 文件顶部的以下常量：

*   `MODEL_NAME`: 指定要使用的预训练语言模型的路径或 Hugging Face Hub 上的名称 (例如: `"/home/tom/fssd/model/Qwen2.5-7B-Instruct"`)。
*   `JSON_PATH`: 输入的 JSON 文件的路径，其中包含要计算 PPL 的对话数据 (例如: `"/home/tom/fssd/yourbench/chemistry_train/rag_reading_5377_claude_sft.json"`)。
*   `SAVE_PATH`: 输出 JSON 文件的保存路径，该文件将包含原始数据以及计算出的 PPL 值 (例如: `"/home/tom/fssd/yourbench/chemistry_train/rag_reading_5377_claude_sft_ppl.json"`)。
*   `BATCH_SIZE`: 批处理大小。根据你的 GPU 显存调整此值。如果遇到显存不足 (Out Of Memory, OOM) 错误，请减小此值 (例如: `1`)。

## 输入数据格式

输入的 JSON 文件 (`JSON_PATH`) 应包含一个列表 (list)，列表中的每个元素是一个字典 (dict)，代表一个对话样本。每个样本字典必须包含一个名为 `conversations` 的键，其值是一个包含两个元素的列表：

1.  第一个元素代表用户的发言，格式为 `{'role': 'user', 'value': '用户的问题或指令'}`。
2.  第二个元素代表助手的回答，格式为 `{'role': 'assistant', 'value': '助手的回答内容'}`。

示例：

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "value": "你好，请介绍一下你自己。"
      },
      {
        "role": "assistant",
        "value": "我是一个大型语言模型..."
      }
    ],
    "other_key": "其他信息..."
  },
  {
    "conversations": [
      {
        "role": "user",
        "value": "解释一下什么是困惑度？"
      },
      {
        "role": "assistant",
        "value": "困惑度是衡量概率模型或概率分布预测样本好坏程度的一种指标..."
      }
    ]
  }
]
```

## 运行

配置好脚本中的常量后，直接运行脚本：

```bash
python ppl.py
```

脚本将显示处理进度，并在完成后打印总耗时和输出文件的路径。

## 输出

脚本执行完毕后，将在 `SAVE_PATH` 指定的位置生成一个新的 JSON 文件。该文件的内容与输入文件类似，但每个对话样本字典中会额外添加一个 `ppl` 键，其值为该样本助手回答部分的困惑度分数。

示例输出 (部分)：

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "value": "你好，请介绍一下你自己。"
      },
      {
        "role": "assistant",
        "value": "我是一个大型语言模型..."
      }
    ],
    "other_key": "其他信息...",
    "ppl": 15.234
  },
  // ... 其他样本
]
```

## 注意事项

*   脚本默认使用 CUDA 设备 0 (`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`)。如果需要使用其他 GPU 或 CPU，请修改此行。
*   脚本使用 `torch.bfloat16` 进行混合精度计算。请确保你的 GPU 支持此数据类型，或根据需要修改 `torch.amp.autocast` 的 `dtype` 参数。
*   如果助手响应为空或无效，PPL 值可能为 `inf` 或 `NaN`。
*   计算 PPL 只针对 `conversations` 列表中的第二个元素 (助手的响应)。 
