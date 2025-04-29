#!/usr/bin/env python3
import os, math, json, time
from typing import List, Dict
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

MODEL_NAME = "/home/tom/fssd/model/Qwen2.5-7B-Instruct"
JSON_PATH  = "/home/tom/fssd/yourbench/chemistry_train/rag_reading_5377_claude_sft.json"
SAVE_PATH  = "/home/tom/fssd/yourbench/chemistry_train/rag_reading_5377_claude_sft_ppl.json"
BATCH_SIZE = 1   # 如仍 OOM 再调小

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
).eval()

def build_batch(batch_items: List[Dict]) -> Dict[str, torch.Tensor]:
    """build batch for ppl calculation"""
    texts, assistant_lens = [], []
    for idx, itm in enumerate(batch_items):
        user_prompt, assistant_prompt = itm['conversations'][0]['value'], itm['conversations'][1]['value']
        messages = [{"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prompt}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        asst_tok_len = len(tokenizer.encode(assistant_prompt, add_special_tokens=False))
        if asst_tok_len <= 0:
            print(f"Warning: Item {idx} in batch has assistant length <= 0 ({asst_tok_len}). Assistant content: '{assistant_prompt[:50]}...'")
        assistant_lens.append(asst_tok_len)

    tok = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids, attn_mask = tok.input_ids, tok.attention_mask
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100
    for i, asst_len in enumerate(assistant_lens):
        seq_len = attn_mask[i].sum().item()
        labels[i, : seq_len - asst_len] = -100
    return dict(input_ids=input_ids, attention_mask=attn_mask, labels=labels)

@torch.inference_mode()
def batch_ppl(tensors: Dict[str, torch.Tensor]) -> List[float]:
    tensors = {k: v.to(model.device) for k, v in tensors.items()}
    logits  = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]).logits

    # shift
    labels  = tensors["labels"][:, 1:]
    logits  = logits[:, :-1, :]
    vocab_size = logits.size(-1)

    # Calculate the mask for valid (non -100) labels
    valid = labels != -100

    # === Pre-gather checks ===
    if not labels.dtype == torch.long:
        print(f"Warning: labels dtype is {labels.dtype}, expected torch.long")
        labels = labels.long() # Attempt to cast

    # Check indices specifically for values used in gather (i.e., not -100)
    active_labels = labels[valid]
    if active_labels.numel() > 0: # Only check if there are labels to gather
        min_label, max_label = active_labels.min(), active_labels.max()
        if min_label < 0 or max_label >= vocab_size:
            print(f"Error: Invalid label index detected before gather! Min: {min_label.item()}, Max: {max_label.item()}, Vocab Size: {vocab_size}")
            # Returning NaNs for the batch as gather would fail
            return [float("nan")] * labels.size(0)
    # =========================

    # Clamp labels before gather to avoid using -100 as index
    # The result for -100 positions will be masked later, so the gathered value doesn't matter.
    labels_for_gather = labels.clone()
    labels_for_gather[labels == -100] = 0 # Replace -100 with a valid index (e.g., 0)

    # NLL  (memory-friendly)
    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, labels_for_gather.unsqueeze(-1)).squeeze(-1)   # [B, L-1]
    nll.masked_fill_(~valid, 0.0)

    ppl = []
    for i in range(nll.size(0)):
        # Use the valid mask derived from the shifted labels
        item_valid_indices = valid[i]
        if not item_valid_indices.any(): # Handle case with no valid labels for this item
            # Assign a high PPL or NaN if assistant response was empty or fully masked
            # Check if assistant_lens[i] was <= 0, maybe assign NaN instead of inf
            if i < len(tensors.get('assistant_original_lens', [])) and tensors['assistant_original_lens'][i] <= 0:
                 ppl.append(float("nan"))
            else:
                 ppl.append(float("inf")) # Or float("nan") if preferred for masked prompts too
            continue

        loss = nll[i][item_valid_indices].mean()
        ppl.append(math.exp(loss.item()) if torch.isfinite(loss) else float("nan"))
    return ppl

def main():
    with open(JSON_PATH) as f: data = json.load(f)
    print("样本数:", len(data))
    t0 = time.time()
    for start in tqdm(range(0, len(data), BATCH_SIZE), desc="Computing PPL"):
        batch = data[start:start+BATCH_SIZE]
        ppls  = batch_ppl(build_batch(batch))
        for j, ppl in enumerate(ppls):
            data[start+j]["ppl"] = ppl
    print(f"耗时 {time.time()-t0:.1f}s")

    with open(SAVE_PATH, "w") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print("done ->", SAVE_PATH)

if __name__ == "__main__":
    main()
