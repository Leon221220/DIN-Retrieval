# -*- coding: utf-8 -*-
"""
Dataset validators for DIN-ICL + vLLM pipeline
==============================================
Unifies:
- prompt building (few-shot + query)
- gold label reading / normalization
- model output parsing
- metric aggregation

Usage (in main):
----------------
from dataset_validators import get_validator

validator = get_validator(task_name=ARGS.task_name)
prompt = validator.build_prompt(tokenizer, query_context, question, exemplars, icl_shots=ARGS.icl_shots)
pred_label = validator.parse_prediction(generation_text)
gold_label  = validator.get_gold_label(example)
is_correct  = validator.is_correct(pred_label, gold_label)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Common base + utilities
# -------------------------
class BaseValidator:
    name = "base"

    # Default system instruction; subclasses may override
    SYSTEM_INSTR = (
        "You are a careful reasoner. Think step by step concisely. "
        "Then on a new line, output exactly one line: Final answer: <LABEL>."
    )

    def system_instruction(self) -> str:
        return self.SYSTEM_INSTR

    def exemplar_to_block(self, ex: Dict[str, Any]) -> str:
        """How an exemplar is rendered in the prompt few-shot block."""
        # Default: show Question + Reasoning + Final
        ctx = ex.get("context") or ex.get("passage") or ex.get("text") or ""
        q   = ex.get("question") or ex.get("input") or ""
        rationale = ex.get("rationale") or ex.get("reasoning") or ex.get("cot") or ""
        ans = self.get_gold_label(ex)
        final = self.format_final(ans)
        parts = []
        if ctx:
            parts += [f"Context:\n{ctx}"]
        parts += [f"Question:\n{q}"]
        if rationale:
            parts += [f"Reasoning:\n{rationale}"]
        parts += [f"{final}"]
        return "\n".join(parts)

    def build_messages(self, context: str, question: str, options: Optional[List[str]]=None, icl_header: str=""):
        """Return chat messages list for tokenizer chat template."""
        user_content = ""
        if icl_header:
            user_content += icl_header.strip() + "\n------\n"
        if context:
            user_content += f"Context:\n{context}\n\n"
        user_content += f"Question:\n{question}\n\n"
        if options:
            user_content += f"Options:\n" + "\n".join(options) + "\n\n"
        user_content += "Reasoning:"
        return [
            {"role": "system", "content": self.system_instruction()},
            {"role": "user",   "content": user_content},
        ]

    # -------- dataset hooks to override --------
    def default_options(self, ex: Dict[str, Any]) -> Optional[List[str]]:
        return None

    def normalize_label(self, y: Any) -> Any:
        return y

    def get_gold_label(self, ex: Dict[str, Any]) -> Any:
        for k in ["answer", "label", "gold", "target", "output"]:
            if k in ex:
                return self.normalize_label(ex[k])
        return None

    def parse_prediction(self, text: str) -> Any:
        """Extract the model final label from raw generation."""
        # default: Final answer: <anything after colon>
        t = text.strip()
        m = re.search(r"final\s*answer\s*:\s*(.+)$", t, re.IGNORECASE | re.MULTILINE)
        if not m:
            return None
        return self.normalize_label(m.group(1))

    def is_correct(self, pred: Any, gold: Any) -> Optional[bool]:
        if gold is None or pred is None:
            return None
        return pred == gold

    def format_final(self, y: Any) -> str:
        return f"Final answer: {y}"

    # -------- top-level API --------
    def build_prompt(self, tokenizer, query_context: str, question: str,
                     exemplars: List[Dict[str, Any]], icl_shots: int = 0, options: Optional[List[str]]=None) -> str:
        fs = ""
        if icl_shots >= 1 and exemplars:
            fs = "\n------\n".join([self.exemplar_to_block(e) for e in exemplars])
        msgs = self.build_messages(query_context, question, self.default_options({}), icl_header=fs)
        if hasattr(tokenizer, "apple_chat_template"):
            return tokenizer.apple_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # type: ignore
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# -------------------------
# PrOntoQA (binary A/B)
# -------------------------
class ProntoQAValidator(BaseValidator):
    name = "prontoqa"
    YES_SET = {"yes","y","true","a","correct"}
    NO_SET  = {"no","n","false","b","incorrect"}

    SYSTEM_INSTR = (
        "You are a careful reasoner. Think step by step concisely. "
        "Then on a new line, output exactly: 'Final answer: A' or 'Final answer: B'."
    )

    def default_options(self, ex: Dict[str, Any]) -> Optional[List[str]]:
        return ["A) True", "B) False"]

    def normalize_label(self, y: Any) -> Optional[str]:
        if y is None:
            return None
        s = str(y).strip().lower()
        if s in self.YES_SET: return "A"
        if s in self.NO_SET:  return "B"
        m = re.match(r"^\s*([ab])\b", s)
        if m: return m.group(1).upper()
        if "true" in s:  return "A"
        if "false" in s: return "B"
        return None
    
class FolioValidator(BaseValidator):
    name = "folio"
    YES_SET = {"yes","y","true","a","correct"}
    NO_SET  = {"no","n","false","b","incorrect"}
    UN_SET ={"uncertain", "u", "c", "unknow"}

    SYSTEM_INSTR = (
        "You are a careful reasoner. Think step by step concisely. "
        "Then on a new line, output exactly: 'Final answer: A' or 'Final answer: B'. or 'Final answer: C'"
    )

    def default_options(self, ex: Dict[str, Any]) -> Optional[List[str]]:
        return ["A) True", "B) False", "C) Uncertain"]

    def normalize_label(self, y: Any) -> Optional[str]:
        if y is None:
            return None
        s = str(y).strip().lower()
        if s in self.YES_SET: return "A"
        if s in self.NO_SET:  return "B"
        if s in self.UN_SET:  return "C"
        m = re.match(r"^\s*([ab])\b", s)
        if m: return m.group(1).upper()
        if "true" in s:  return "A"
        if "false" in s: return "B"
        if "unkonw" in s: return "C"
        return None
    
# -------------------------
# GSM8K（开放数值答案）
# -------------------------
class GSM8KValidator(BaseValidator):
    name = "gsm8k"

    SYSTEM_INSTR = (
        "You are a careful math reasoner. Solve step by step concisely. "
        "Then on a new line, output exactly: 'Final answer: <number>'."
    )

    def normalize_label(self, y: Any) -> Optional[str]:
        if y is None: return None
        s = str(y)
        # strip common wrappers like \boxed{...}, units, commas, spaces
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        s = re.sub(r"[,$ ]", "", s)
        # keep only leading optional sign + digits + optional decimal
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        return m.group(0) if m else None

    def parse_prediction(self, text: str) -> Optional[str]:
        # Prefer "Final answer: xxx"
        m = re.search(r"final\s*answer\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        s = m.group(1) if m else text
        return self.normalize_label(s)

# -------------------------
# CSQA（多选 A-E）
# -------------------------
class CSQAValidator(BaseValidator):
    name = "csqa"

    SYSTEM_INSTR = (
        "You are a careful reasoner. Think briefly. "
        "Then output exactly one line: 'Final answer: A/B/C/D/E'."
    )

    def default_options(self, ex: Dict[str, Any]) -> Optional[List[str]]:
        # If options exist in example, we don't override; main code can list them
        return None

    def normalize_label(self, y: Any) -> Optional[str]:
        if y is None: return None
        s = str(y).strip().upper()
        m = re.match(r"^\s*([A-E])\b", s)
        if m: return m.group(1)
        # some datasets store gold as the full string of the correct option
        if len(s) == 1 and s in "ABCDE": return s
        return None

# -------------------------
# ARC-Challenge（多选 A-D/E）
# -------------------------
class ARCValidator(CSQAValidator):
    name = "arc"

# -------------------------
# MMLU（多选 A-D）
# -------------------------
class MMLUValidator(CSQAValidator):
    name = "mmlu"
    def normalize_label(self, y: Any) -> Optional[str]:
        if y is None: return None
        s = str(y).strip().upper()
        m = re.match(r"^\s*([A-D])\b", s)
        if m: return m.group(1)
        if len(s) == 1 and s in "ABCD": return s
        return None

# -------------------------
# BoolQ（Yes/No）
# -------------------------
class BoolQValidator(ProntoQAValidator):
    name = "boolq"
    # In BoolQ, labels are true/false; reuse A/B mapping via ProntoQAValidator

# -------------------------
# StrategyQA（Yes/No）
# -------------------------
class StrategyQAValidator(ProntoQAValidator):
    name = "strategyqa"

# -------------------------
# Registry
# -------------------------
_REG = {
    ProntoQAValidator.name: ProntoQAValidator,
    GSM8KValidator.name:    GSM8KValidator,
    CSQAValidator.name:     CSQAValidator,
    ARCValidator.name:      ARCValidator,
    MMLUValidator.name:     MMLUValidator,
    BoolQValidator.name:    BoolQValidator,
    StrategyQAValidator.name: StrategyQAValidator,
    FolioValidator.name: FolioValidator
}

def get_validator(task_name: str) -> BaseValidator:
    key = (task_name or "").strip().lower()
    if key in _REG:
        return _REG[key]()
    # fallbacks/aliases
    if key in {"pronto", "prontoqA"}:
        return ProntoQAValidator()
    if key in {"arc-c", "arc_challenge"}:
        return ARCValidator()
    if key in {"folio"}:
        return FolioValidator
    raise ValueError(f"Unknown task_name '{task_name}'. Available: {list(_REG.keys())}")
