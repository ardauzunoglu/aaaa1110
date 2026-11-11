#!/usr/bin/env python3
"""Evaluate pretrained causal LMs on next-token prediction."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_dtype(name: str):
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def normalize_text_row(row: dict, text_key: str) -> str | None:
    value = row.get(text_key)
    if value is None:
        for fallback in ("text", "document", "content"):
            value = row.get(fallback)
            if isinstance(value, str):
                break
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def load_json_records(path: Path, text_key: str) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[dict] = []
    if isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        for key in ("rows", "records", "data", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                rows = [x for x in value if isinstance(x, dict)]
                break
        if not rows:
            rows = [data]
    out: List[str] = []
    for row in rows:
        text = normalize_text_row(row, text_key)
        if text is not None:
            out.append(text)
    return out


def load_jsonl_records(path: Path, text_key: str) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            text = normalize_text_row(row, text_key)
            if text is not None:
                out.append(text)
    return out


def load_csv_records(path: Path, text_key: str) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            text = normalize_text_row(row, text_key)
            if text is not None:
                out.append(text)
    return out


def load_text_corpus(path: Path, text_key: str) -> List[str]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix in {".txt", ".jsonl", ".json", ".csv"}
        )
    else:
        raise FileNotFoundError(f"Input path not found: {path}")

    texts: List[str] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".txt":
            parsed = [x.strip() for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
        elif suffix == ".json":
            parsed = load_json_records(file, text_key)
        elif suffix == ".jsonl":
            parsed = load_jsonl_records(file, text_key)
        elif suffix == ".csv":
            parsed = load_csv_records(file, text_key)
        else:
            parsed = []
        if not parsed:
            logger.warning("No usable text rows found in %s", file)
            continue
        texts.extend(parsed)
        logger.info("Loaded %d rows from %s", len(parsed), file.name)

    if not texts:
        raise ValueError(f"No usable text records found in {path}")
    return texts


def docs_to_token_ids(
    texts: Iterable[str],
    tokenizer,
    add_eos_between_docs: bool,
    max_docs: int | None,
    max_tokens: int | None,
    min_doc_tokens: int | None,
    max_doc_tokens: int | None,
) -> tuple[List[List[int]], List[str], dict[str, int]]:
    docs_token_ids: List[List[int]] = []
    docs_texts: List[str] = []
    eos_id = tokenizer.eos_token_id
    docs_used = 0
    total_tokens = 0
    truncated = False
    docs_skipped_empty = 0
    docs_skipped_min_tokens = 0
    docs_skipped_max_tokens = 0
    for text in texts:
        if max_docs is not None and docs_used >= max_docs:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            docs_skipped_empty += 1
            continue
        if min_doc_tokens is not None and len(ids) < min_doc_tokens:
            docs_skipped_min_tokens += 1
            continue
        if max_doc_tokens is not None and len(ids) > max_doc_tokens:
            docs_skipped_max_tokens += 1
            continue
        if add_eos_between_docs and eos_id is not None:
            ids = ids + [eos_id]
        if max_tokens is not None and total_tokens >= max_tokens:
            break
        remaining = None if max_tokens is None else max_tokens - total_tokens
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and len(ids) > remaining:
            ids = ids[:remaining]
            truncated = True
        if not ids:
            break
        docs_token_ids.append(ids)
        docs_texts.append(text)
        total_tokens += len(ids)
        docs_used += 1
        if truncated:
            break
    logger.info(
        "Tokenized %d docs into %d tokens%s (skipped: empty=%d min_tokens=%d max_tokens=%d)",
        docs_used,
        total_tokens,
        " (truncated by --max-tokens)" if truncated else "",
        docs_skipped_empty,
        docs_skipped_min_tokens,
        docs_skipped_max_tokens,
    )
    stats = {
        "docs_tokenized": docs_used,
        "docs_skipped_empty": docs_skipped_empty,
        "docs_skipped_min_tokens": docs_skipped_min_tokens,
        "docs_skipped_max_tokens": docs_skipped_max_tokens,
    }
    return docs_token_ids, docs_texts, stats


def evaluate_docs(
    model,
    tokenizer,
    device: str,
    docs_token_ids: List[List[int]],
    docs_texts: List[str],
    batch_size: int,
    progress_every: int,
) -> dict:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    docs_for_eval = [(text, ids) for text, ids in zip(docs_texts, docs_token_ids) if len(ids) >= 2]
    skipped_too_short = len(docs_token_ids) - len(docs_for_eval)
    logger.info(
        "Prepared %d docs for isolated evaluation (%d docs too short for next-token targets)",
        len(docs_for_eval),
        skipped_too_short,
    )

    if not docs_for_eval:
        raise ValueError("No documents had at least 2 tokens for next-token evaluation.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            pad_id = eos_id
            logger.warning("Tokenizer has no pad token; using eos_token_id=%d for padding", pad_id)
        else:
            pad_id = 0
            logger.warning("Tokenizer has neither pad nor eos token; using 0 for padding")

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    document_to_avg_loss: dict[str, float] = {}
    total_batches = math.ceil(len(docs_for_eval) / batch_size)

    for start in range(0, len(docs_for_eval), batch_size):
        batch = docs_for_eval[start : start + batch_size]
        batch_texts = [text for text, _ in batch]
        batch_docs = [ids for _, ids in batch]
        input_rows = [x[:-1] for x in batch_docs]
        label_rows = [x[1:] for x in batch_docs]
        max_len = max(len(x) for x in input_rows)
        padded_inputs: List[List[int]] = []
        padded_labels: List[List[int]] = []
        padded_mask: List[List[int]] = []
        for input_ids, labels in zip(input_rows, label_rows):
            pad_len = max_len - len(input_ids)
            padded_inputs.append(input_ids + [pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_mask.append([1] * len(input_ids) + [0] * pad_len)

        input_ids_t = torch.tensor(padded_inputs, dtype=torch.long, device=device)
        labels_t = torch.tensor(padded_labels, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(padded_mask, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attention_mask_t).logits
            vocab_size = logits.shape[-1]
            token_losses = F.cross_entropy(
                logits.view(-1, vocab_size), labels_t.view(-1), ignore_index=-100, reduction="none"
            ).view_as(labels_t)
            preds = torch.argmax(logits, dim=-1)
            valid_mask = labels_t != -100
            correct = ((preds == labels_t) & valid_mask).sum()
            num_tokens = valid_mask.sum()
            per_doc_tokens = valid_mask.sum(dim=1)
            per_doc_nll = (token_losses * valid_mask).sum(dim=1)
            loss = per_doc_nll.sum()

        for text, doc_nll, doc_tokens in zip(batch_texts, per_doc_nll, per_doc_tokens):
            document_to_avg_loss[text] = float((doc_nll / doc_tokens).item())

        total_nll += float(loss.item())
        total_tokens += int(num_tokens.item())
        total_correct += int(correct.item())

        batch_idx = (start // batch_size) + 1
        if progress_every > 0 and batch_idx % progress_every == 0:
            mean_nll_so_far = total_nll / total_tokens
            ppl_so_far = float(math.exp(mean_nll_so_far))
            acc_so_far = total_correct / total_tokens
            logger.info(
                "Progress: %d/%d batches | ppl=%.4f mean_nll=%.6f top1_acc=%.4f tokens=%d",
                batch_idx,
                total_batches,
                ppl_so_far,
                mean_nll_so_far,
                acc_so_far,
                total_tokens,
            )

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated.")

    mean_nll = total_nll / total_tokens
    perplexity = float(math.exp(mean_nll))
    accuracy = total_correct / total_tokens
    return {
        "num_docs_evaluated": len(docs_for_eval),
        "num_docs_skipped_too_short": skipped_too_short,
        "num_batches": total_batches,
        "num_tokens": total_tokens,
        "total_correct": total_correct,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "top1_accuracy": accuracy,
        "document_to_avg_loss": document_to_avg_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to a text dataset file or directory. Repeat --input to evaluate multiple corpora together.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Text column/key for .jsonl/.json/.csv (fallbacks: text, document, content).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name/path for AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name/path. Defaults to --model.",
    )
    parser.add_argument(
        "--output",
        default="next_token_eval_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of documents per forward pass.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of documents loaded.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional cap on total tokenized corpus length.",
    )
    parser.add_argument(
        "--min-doc-tokens",
        type=int,
        default=None,
        help="Optional minimum document length in tokens (before optional EOS append).",
    )
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=None,
        help="Optional maximum document length in tokens (before optional EOS append).",
    )
    parser.add_argument(
        "--no-eos-between-docs",
        action="store_true",
        help="Disable appending EOS token at the end of each document.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Log progress every N batches (0 disables).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.min_doc_tokens is not None and args.min_doc_tokens <= 0:
        raise ValueError("--min-doc-tokens must be > 0")
    if args.max_doc_tokens is not None and args.max_doc_tokens <= 0:
        raise ValueError("--max-doc-tokens must be > 0")
    if (
        args.min_doc_tokens is not None
        and args.max_doc_tokens is not None
        and args.min_doc_tokens > args.max_doc_tokens
    ):
        raise ValueError("--min-doc-tokens must be <= --max-doc-tokens")

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    logger.info("Using device=%s dtype=%s", device, args.dtype)

    input_paths = [Path(path_str) for path_str in args.input]
    loaded_sources = []
    source_doc_counts_raw: dict[str, int] = {}
    source_doc_counts_used: dict[str, int] = {}
    for input_path in input_paths:
        logger.info("Loading corpus from %s", input_path)
        loaded = load_text_corpus(input_path, text_key=args.text_key)
        source_doc_counts_raw[str(input_path)] = len(loaded)
        if args.max_docs is not None:
            before = len(loaded)
            loaded = loaded[: args.max_docs]
            logger.info(
                "Applied --max-docs=%d to %s (using %d/%d docs)",
                args.max_docs,
                input_path,
                len(loaded),
                before,
            )
        source_doc_counts_used[str(input_path)] = len(loaded)
        loaded_sources.append((input_path, loaded))

    logger.info(
        "Loaded %d total documents from %d input source(s)",
        sum(len(texts) for _, texts in loaded_sources),
        len(loaded_sources),
    )

    tokenizer_name = args.tokenizer or args.model
    logger.info("Loading tokenizer from %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    logger.info("Loading model from %s", args.model)
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = os.environ.get("HF_ATTN_IMPLEMENTATION", "flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()

    per_input_results = []
    total_docs_tokenized = 0
    total_docs_skipped_empty = 0
    total_docs_skipped_min_tokens = 0
    total_docs_skipped_max_tokens = 0
    total_docs_evaluated = 0
    total_docs_skipped_too_short = 0
    total_batches = 0
    total_tokens = 0
    total_nll = 0.0
    total_correct = 0
    document_to_avg_loss: dict[str, float] = {}

    for input_path, source_texts in loaded_sources:
        logger.info("Running next-token eval for source=%s", input_path)
        docs_token_ids, docs_texts, tokenization_stats = docs_to_token_ids(
            texts=source_texts,
            tokenizer=tokenizer,
            add_eos_between_docs=not args.no_eos_between_docs,
            max_docs=None,
            max_tokens=args.max_tokens,
            min_doc_tokens=args.min_doc_tokens,
            max_doc_tokens=args.max_doc_tokens,
        )
        source_metrics = evaluate_docs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            docs_token_ids=docs_token_ids,
            docs_texts=docs_texts,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )

        total_docs_tokenized += tokenization_stats["docs_tokenized"]
        total_docs_skipped_empty += tokenization_stats["docs_skipped_empty"]
        total_docs_skipped_min_tokens += tokenization_stats["docs_skipped_min_tokens"]
        total_docs_skipped_max_tokens += tokenization_stats["docs_skipped_max_tokens"]
        total_docs_evaluated += source_metrics["num_docs_evaluated"]
        total_docs_skipped_too_short += source_metrics["num_docs_skipped_too_short"]
        total_batches += source_metrics["num_batches"]
        total_tokens += source_metrics["num_tokens"]
        total_nll += source_metrics["total_nll"]
        total_correct += source_metrics["total_correct"]
        document_to_avg_loss.update(source_metrics.pop("document_to_avg_loss"))

        per_input_results.append(
            {
                "input_path": str(input_path),
                "docs_loaded_raw": source_doc_counts_raw[str(input_path)],
                "docs_loaded_used": source_doc_counts_used[str(input_path)],
                "tokenization": tokenization_stats,
                "metrics": source_metrics,
            }
        )

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated across inputs.")

    mean_nll = total_nll / total_tokens
    metrics = {
        "num_docs_evaluated": total_docs_evaluated,
        "num_docs_skipped_too_short": total_docs_skipped_too_short,
        "num_batches": total_batches,
        "num_tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": float(math.exp(mean_nll)),
        "top1_accuracy": total_correct / total_tokens,
    }

    output = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(input_paths[0]) if len(input_paths) == 1 else None,
            "input_paths": [str(path) for path in input_paths],
            "input_source_doc_counts_raw": source_doc_counts_raw,
            "input_source_doc_counts_used": source_doc_counts_used,
            "text_key": args.text_key,
            "evaluation_mode": "per_document",
            "model": args.model,
            "tokenizer": tokenizer_name,
            "device": device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_docs": args.max_docs,
            "max_tokens": args.max_tokens,
            "min_doc_tokens": args.min_doc_tokens,
            "max_doc_tokens": args.max_doc_tokens,
            "add_eos_between_docs": not args.no_eos_between_docs,
            "docs_tokenized": total_docs_tokenized,
            "docs_skipped_empty": total_docs_skipped_empty,
            "docs_skipped_min_tokens": total_docs_skipped_min_tokens,
            "docs_skipped_max_tokens": total_docs_skipped_max_tokens,
            "docs_evaluated": metrics["num_docs_evaluated"],
            "docs_skipped_too_short": metrics["num_docs_skipped_too_short"],
            "results_are_split_by_input": len(input_paths) > 1,
        },
        "metrics": metrics,
        "per_input_results": per_input_results,
        "document_to_avg_loss": document_to_avg_loss,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation JSON to %s", out_path)
    logger.info(
        "Next-token metrics: ppl=%.4f mean_nll=%.6f top1_acc=%.4f tokens=%d",
        metrics["perplexity"],
        metrics["mean_nll"],
        metrics["top1_accuracy"],
        metrics["num_tokens"],
    )


if __name__ == "__main__":
    main()
