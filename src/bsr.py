import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

@torch.no_grad()
def _encode_tokens(texts, tok, model, max_len=1024, device="cuda"):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    out = model(**enc, output_hidden_states=True, return_dict=True)
    # 取最后一层的 token 表征（B, L, D），mask掉padding
    hs = out.hidden_states[-1]  # (B, L, D)
    mask = enc["attention_mask"].bool()  # (B, L)
    return hs, mask  # 交给下游按每条样本切分

def _bsr_recall_one(query_tok_emb, query_mask, cand_tok_emb, cand_mask):
    # query: (Lq, D), cand: (Lc, D)
    q = query_tok_emb[query_mask]          # (Lq, D)
    c = cand_tok_emb[cand_mask]            # (Lc, D)
    if q.size(0) == 0 or c.size(0) == 0:
        return 0.0
    q = torch.nn.functional.normalize(q, dim=-1)
    c = torch.nn.functional.normalize(c, dim=-1)
    # 相似度 (Lq, Lc)
    sim = q @ c.T
    # BSR-Recall: 对每个query token取 cand 里最高匹配，再对所有query token取均值
    # 可加idf权重，这里给出无idf的简单实现
    per_q = sim.max(dim=1).values  # (Lq,)
    return float(per_q.mean().item())

class BSRScorer:
    def __init__(self, model_name="roberta-large", device=None, max_len=1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_len = max_len

    @torch.no_grad()
    def score_queries_against_pool(self, queries, pool_texts):
        # 预编码（批处理可自行扩展；这里为简洁起见按全量）
        q_hs, q_mask = _encode_tokens(queries, self.tok, self.model, self.max_len, self.device)
        p_hs, p_mask = _encode_tokens(pool_texts, self.tok, self.model, self.max_len, self.device)
        scores = []
        for i in range(len(queries)):
            qi_emb, qi_mask = q_hs[i], q_mask[i]
            # 对所有pool逐个算 BSR
            s = [ _bsr_recall_one(qi_emb, qi_mask, p_hs[j], p_mask[j]) for j in range(len(pool_texts)) ]
            scores.append(s)
        return scores  # List[List[float]]: 每个query对每个pool的BSR

    
@torch.no_grad()
def greedy_set_bsr_select(
    query_text, pool_texts, scorer: BSRScorer, topk=2
):
    # 编码 query 与所有候选
    q_hs, q_mask = _encode_tokens([query_text], scorer.tok, scorer.model, scorer.max_len, scorer.device)
    p_hs, p_mask = _encode_tokens(pool_texts, scorer.tok, scorer.model, scorer.max_len, scorer.device)

    q_emb, q_m = q_hs[0], q_mask[0]              # (Lq,D), (Lq,)
    q_tok = torch.nn.functional.normalize(q_emb[q_m], dim=-1)   # (Lq,D)
    Lq = q_tok.size(0)
    if Lq == 0:
        return list(range(min(topk, len(pool_texts))))

    # 预先计算每个候选对每个query token的最大相似（=该候选对该token的覆盖度）
    cand_cov = []   # list of tensor (Lq,)
    for j in range(len(pool_texts)):
        c_emb, c_m = p_hs[j], p_mask[j]
        if c_m.sum() == 0:
            cand_cov.append(torch.zeros(Lq, device=q_tok.device))
            continue
        c_tok = torch.nn.functional.normalize(c_emb[c_m], dim=-1)     # (Lc,D)
        sim = q_tok @ c_tok.T                                         # (Lq,Lc)
        cov = sim.max(dim=1).values                                   # (Lq,)
        cand_cov.append(cov)

    # 贪心：每次选能最大提升 sum(max_coverage) 的候选
    selected = []
    cur_cov = torch.zeros(Lq, device=q_tok.device)
    remaining = set(range(len(pool_texts)))

    for _ in range(min(topk, len(pool_texts))):
        best_gain, best_j = -1e9, None
        for j in remaining:
            new_cov = torch.maximum(cur_cov, cand_cov[j])
            gain = float(new_cov.sum().item() - cur_cov.sum().item())
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None: break
        selected.append(best_j)
        cur_cov = torch.maximum(cur_cov, cand_cov[best_j])
        remaining.remove(best_j)

    return selected
