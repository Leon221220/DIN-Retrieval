#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click evaluation for Meta-Llama-3.1-8B-Instruct on PrOntoQA-style data
- Input JSON/JSONL fields: {id, context, question, answer, options}
  * answer must be "A" or "B" (A=True, B=False) or boolean/0-1 (will map to A/B)
- Backends:
  * HF (transformers generate, left padding)  --backend hf
  * vLLM (vllm.LLM.generate)                  --backend vllm
- Zero-shot CoT with final line: `Final answer: A` or `Final answer: B`
- Optional 2-shot exemplars (--icl_shots 2)
- Saves predictions to JSONL/CSV if specified
"""

import argparse
import csv
import json
import os
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================
# Few-shot exemplars (A/B final)
# =============================
EX1 = """Given a problem statement as contexts, the task is to answer a logical reasoning question. 
------
Context:
Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. Dumpuses are not wooden. Dumpuses are wumpuses. Wumpuses are red. Every wumpus is an impus. Each impus is opaque. Impuses are tumpuses. Numpuses are sour. Tumpuses are not sour. Tumpuses are vumpuses. Vumpuses are earthy. Every vumpus is a zumpus. Zumpuses are small. Zumpuses are rompuses. Max is a yumpus.

Question:
Is the following statement true or false? Max is sour.

Options:
A) True
B) False

Reasoning:
Max is a yumpus. Each yumpus is a dumpus. So Max is a dumpus. Dumpuses are wumpuses. So Max is a wumpus. Every wumpus is an impus. So Max is an impus. Impuses are tumpuses. So Max is a tumpus. Tumpuses are not sour. Therefore Max is not sour.

Final answer: B"""

EX2 = """Given a problem statement as contexts, the task is to answer a logical reasoning question. 
------
Context:
Every tumpus is not angry. Tumpuses are rompuses. Every numpus is not bright. Rompuses are not luminous. Rompuses are yumpuses. Yumpuses are transparent. Yumpuses are zumpuses. Each zumpus is not bitter. Zumpuses are impuses. Impuses are red. Each impus is a dumpus. Every dumpus is happy. Each dumpus is a vumpus. Vumpuses are bright. Every vumpus is a jompus. Jompuses are large. Each jompus is a wumpus. Stella is a yumpus.

Question:
Is the following statement true or false? Stella is bright.

Options:
A) True
B) False

Reasoning:
Stella is a yumpus. Yumpuses are zumpuses, so Stella is a zumpus. Zumpuses are impuses, so Stella is an impus. Each impus is a dumpus, thus Stella is a dumpus. Every dumpus is a vumpus; vumpuses are bright. Therefore Stella is bright.

Final answer: A"""

YES_SET = {"yes", "y", "true", "correct"}
NO_SET  = {"no", "n", "false", "incorrect"}

# -----------------------------
# Utils
# -----------------------------
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_final_ab(s: str) -> Optional[str]:
    """
    Parse final A/B:
    1) strict: Final answer: A/B
    2) fallback: true/false or yes/no -> map to A/B
    """
    t = norm_text(s)
    m = re.search(r"final\s*answer\s*:\s*([ab])\b", t)
    if m:
        return m.group(1).upper()
    tf = re.search(r"\b(true|false)\b", t)
    if tf:
        return "A" if tf.group(1) == "true" else "B"
    yes_m = re.search(r"\b(yes|true|correct)\b", t)
    no_m  = re.search(r"\b(no|false|incorrect)\b", t)
    if yes_m and (not no_m or yes_m.start() < no_m.start()):
        return "A"
    if no_m and (not yes_m or no_m.start() < yes_m.start()):
        return "B"
    return None

def ab_to_bool(label_ab: Optional[str]) -> Optional[bool]:
    if label_ab is None: return None
    if label_ab.upper() == "A": return True
    if label_ab.upper() == "B": return False
    return None

# -----------------------------
# Prompt building
# -----------------------------
SYSTEM_INSTR = (
    "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
)

def build_messages_from_fields(context: str, question: str, options: List[str], icl_shots: int = 0) -> List[Dict[str, str]]:
    fewshot_block = ""
    if icl_shots >= 2:
        fewshot_block = EX1 + "\n------\n" + EX2 + "\n------\n"
    if not options or not isinstance(options, list) or len(options) < 2:
        options = ["A) True", "B) False"]
    options_str = "\n".join(options[:2])
    user_content = (
        f"{fewshot_block}"
        f"------\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Reasoning:"
    )
    return [
        {"role": "system", "content": SYSTEM_INSTR},
        {"role": "user", "content": user_content},
    ]

def render_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Unify templating across backends for identical prompts
    if hasattr(tokenizer, "apple_chat_template"):
        return tokenizer.apple_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -----------------------------
# Data loading
# -----------------------------
def load_json_or_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Expect records: {id, context, question, answer, options}
    'answer' can be "A"/"B" or true/false or 0/1
    Output: {"id","context","question","options","gold_ab","gold_bool"}
    """
    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        else:
            for _, v in data.items():
                if isinstance(v, list):
                    items = v
                    break
    else:
        raise ValueError("Dataset file must be .json or .jsonl")

    out = []
    for ex in items:
        ctx = ex.get("context")
        qst = ex.get("question")
        ans = ex.get("answer")
        opts = ex.get("options", ["A) True", "B) False"])
        gold_ab = None
        if isinstance(ans, str):
            t = norm_text(ans)
            if t in {"a", "true"}: gold_ab = "A"
            elif t in {"b", "false"}: gold_ab = "B"
        elif isinstance(ans, bool):
            gold_ab = "A" if ans else "B"
        elif isinstance(ans, (int, float)):
            gold_ab = "A" if int(ans) == 1 else "B"
        if ctx and qst and gold_ab in {"A","B"}:
            out.append({
                "id": ex.get("id"),
                "context": ctx,
                "question": qst,
                "options": opts if isinstance(opts, list) and len(opts)>=2 else ["A) True", "B) False"],
                "gold_ab": gold_ab,
                "gold_bool": (gold_ab == "A"),
            })
    if limit is not None:
        out = out[:limit]
    return out

# -----------------------------
# Backends
# -----------------------------
def run_hf(model, tokenizer, device, batch_prompts, max_new_tokens, temperature, top_p):
    # Left-padding batch encode (we already rendered strings)
    enc = tokenizer(
        batch_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    texts = []
    for i in range(outputs.size(0)):
        prompt_len = int(attention_mask[i].sum().item())
        gen_ids = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
    return texts

def run_vllm(llm, batch_prompts, max_new_tokens, temperature, top_p):
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outs = llm.generate(batch_prompts, sp)
    texts = []
    for o in outs:
        # pick the first candidate
        texts.append(o.outputs[0].text if o.outputs else "")
    return texts

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                        help="Choose transformers HF or vLLM backend.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to .json/.jsonl with fields {id, context, question, answer, options}.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default=None, help="e.g., cuda:0 or cpu; only for HF backend")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager"],
                        help="Only for HF backend.")
    parser.add_argument("--icl_shots", type=int, default=0, choices=[0, 2],
                        help="Number of in-context exemplars to prepend (0 or 2).")
    # vLLM-specific
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length.")
    parser.add_argument("--gpu_mem", type=float, default=0.9, help="vLLM gpu_memory_utilization (e.g., 0.9).")
    # saving
    parser.add_argument("--save_jsonl", type=str, default=None,
                        help="If set, save per-example predictions to this JSONL file.")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="If set, also save per-example predictions to this CSV file.")
    args = parser.parse_args()

    # Load data
    data = load_json_or_jsonl(args.dataset, limit=args.limit)
    print(f"[Info] Loaded {len(data)} examples from {args.dataset}")

    # Common tokenizer (for chat template rendering to string; works for both backends)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Backend init
    llm = None
    model = None
    device = None
    if args.backend == "hf":
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        print(f"[Info] HF backend on {device}, dtype={args.dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
            attn_implementation=args.attn_implementation,
        ).to(device)
        model.eval()
    else:
        # vLLM
        from vllm import LLM
        vllm_kwargs = dict(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tp,
            trust_remote_code=True,
            dtype='bfloat16'
        )
        if args.max_model_len is not None:
            vllm_kwargs["max_model_len"] = args.max_model_len
        if args.gpu_mem is not None:
            vllm_kwargs["gpu_memory_utilization"] = args.gpu_mem
        # dtype: vLLM会自动选；如强制可加 vllm_kwargs["dtype"] = "bfloat16"/"float16"
        print(f"[Info] vLLM backend init: tp={args.tp}, max_model_len={args.max_model_len}, gpu_mem={args.gpu_mem}")
        llm = LLM(**vllm_kwargs)

    # Savers
    jsonl_fp = open(args.save_jsonl, "w", encoding="utf-8") if args.save_jsonl else None
    csv_fp, csv_writer = None, None
    if args.save_csv:
        csv_fp = open(args.save_csv, "w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["idx", "id", "gold_ab", "gold_bool", "pred_text", "pred_ab", "pred_bool", "correct"])

    correct = 0
    total = 0
    global_idx = 0

    # Batched inference
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        prompts = [
            render_chat_prompt(
                tokenizer,
                build_messages_from_fields(ex["context"], ex["question"], ex.get("options", ["A) True", "B) False"]), icl_shots=args.icl_shots)
            )
            for ex in batch
        ]

        if args.backend == "hf":
            gen_texts = run_hf(
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            gen_texts = run_vllm(
                llm=llm,
                batch_prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        # evaluate + save
        for ex, pred_text in zip(batch, gen_texts):
            gold_ab = ex["gold_ab"]
            gold_bool = ex["gold_bool"]

            pred_ab = extract_final_ab(pred_text)
            pred_bool = ab_to_bool(pred_ab)

            is_correct = (pred_bool is not None and pred_bool == gold_bool)
            total += 1
            if is_correct: correct += 1

            rec = {
                "idx": global_idx,
                "id": ex.get("id"),
                "gold_ab": gold_ab,
                "gold_bool": gold_bool,
                "pred_text": pred_text.strip(),
                "pred_ab": pred_ab,
                "pred_bool": pred_bool,
                "correct": bool(is_correct),
            }
            global_idx += 1

            if jsonl_fp:
                jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if csv_writer:
                csv_writer.writerow([
                    rec["idx"], rec["id"], rec["gold_ab"], rec["gold_bool"],
                    rec["pred_text"], rec["pred_ab"], rec["pred_bool"], rec["correct"]
                ])

    acc = correct / max(total, 1)
    print(f"[Result] PrOntoQA (CoT, A/B, backend={args.backend}) Accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"[Config] backend={args.backend}, batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}, icl_shots={args.icl_shots}")

    if jsonl_fp:
        jsonl_fp.close()
        print(f"[Saved] JSONL predictions -> {args.save_jsonl}")
    if csv_fp:
        csv_fp.close()
        print(f"[Saved] CSV predictions   -> {args.save_csv}")

if __name__ == "__main__":
    main()
