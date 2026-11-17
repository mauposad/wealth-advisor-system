"""Run sample evaluation prompts and save inspectable evidence for manual review."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from calculator import route_calculation, format_calculator_block
from citations import audit_citations
from llm_client import answer
from prompts import SYSTEM_PROMPT, build_user_prompt
from rag import Retriever, STUDENT_LOAN_CORPUS_PATTERNS, format_context
from runtime_manifest import write_manifest

load_dotenv()
ROOT = Path(__file__).parent
retriever = Retriever(ROOT / "corpus", include_patterns=STUDENT_LOAN_CORPUS_PATTERNS)
top_k = int(os.getenv("RAG_TOP_K", "4"))
min_retrieval_score = float(os.getenv("RAG_MIN_SCORE", "0.15"))

cases_filename = os.getenv("EVAL_CASES_FILE", "eval_cases.json")
cases_path = ROOT / cases_filename
if not cases_path.is_file():
    raise FileNotFoundError(f"Evaluation cases file not found: {cases_path}")
cases = json.loads(cases_path.read_text(encoding="utf-8"))
manifest, manifest_path = write_manifest(ROOT, SYSTEM_PROMPT, top_k)
results = []

for case in cases:
    query = case["prompt"]
    history = case.get("history", [])
    hits = retriever.search(query, top_k, min_retrieval_score)
    context = format_context(hits)
    calc = route_calculation(query)
    prompt = build_user_prompt(query, context, format_calculator_block(calc), history)

    t0 = time.perf_counter()
    try:
        text, backend, model = answer(SYSTEM_PROMPT, prompt)
        error = None
    except Exception as e:
        text, backend, model, error = "", os.getenv("LLM_BACKEND", "mock"), "unknown", repr(e)
    latency = time.perf_counter() - t0

    retrieved_ids = [h.chunk_id for h in hits]
    results.append({
        **case,
        "answer": text,
        "backend": backend,
        "model": model,
        "retrieved": [h.to_dict() for h in hits],
        "retrieved_ids": retrieved_ids,
        "needed_evidence_present": None,
        "calculator": calc.to_dict() if calc else None,
        "prompt_context": prompt,
        "citation_audit": audit_citations(text, retrieved_ids),
        "latency_seconds": latency,
        "error": error,
    })

payload = {"manifest": manifest, "results": results}
out = ROOT / "logs" / f"{cases_path.stem}_results.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote {len(results)} results from {cases_path.name} to {out}")
print(f"Wrote run manifest to {manifest_path}")
print("Manual review reminder: for each failure, first mark whether the evidence needed for a good answer was present in the retrieved context.")
