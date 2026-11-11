#!/usr/bin/env python3
"""Evaluate multiple-choice bank files with accuracy and loss metrics."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

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


def answer_to_index(value: object, num_choices: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Unsupported answer type: {type(value)}")

    if isinstance(value, int):
        idx = value
        if 0 <= idx < num_choices:
            return idx
        if 1 <= idx <= num_choices:
            return idx - 1
        raise ValueError(f"Answer index out of range: {idx}")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            idx = int(stripped)
            if 0 <= idx < num_choices:
                return idx
            if 1 <= idx <= num_choices:
                return idx - 1
        upper = stripped.upper()
        if len(upper) == 1 and "A" <= upper <= "Z":
            idx = ord(upper) - ord("A")
            if 0 <= idx < num_choices:
                return idx
        raise ValueError(f"Unsupported answer value: {value!r}")

    raise ValueError(f"Unsupported answer type: {type(value)}")


def load_json_records(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("rows", "questions", "by_question", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_jsonl_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_bank(path: Path) -> List[dict]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.name != "manifest.json"
            and p.suffix in {".jsonl", ".json"}
        )
    else:
        raise FileNotFoundError(f"Bank path not found: {path}")

    rows: List[dict] = []
    for file in files:
        if file.suffix == ".jsonl":
            parsed = load_jsonl_records(file)
        else:
            parsed = load_json_records(file)
        if not parsed:
            logger.warning("No rows found in %s", file)
            continue
        rows.extend(parsed)
        logger.info("Loaded %d rows from %s", len(parsed), file.name)

    if not rows:
        raise ValueError(f"No usable records found in {path}")
    return rows


def normalize_rows(rows: Iterable[dict]) -> List[dict]:
    out: List[dict] = []
    skipped = 0
    for i, row in enumerate(rows):
        try:
            question = str(row["question"]).strip()
            choices = [str(x).strip() for x in row["choices"]]
            if not question:
                raise ValueError("Empty question")
            if len(choices) < 2:
                raise ValueError(f"Expected >=2 choices, got {len(choices)}")
            answer_raw = row.get("answer_idx", row.get("answer"))
            answer_idx = answer_to_index(answer_raw, len(choices))
            out.append(
                {
                    "id": row.get("id", f"q-{i:06d}"),
                    "subject": str(row.get("subject", "unknown")),
                    "dataset": str(row.get("dataset", row.get("subject", "unknown"))),
                    "question": question,
                    "choices": choices,
                    "answer_idx": answer_idx,
                }
            )
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping malformed row %d (%s)", i, exc)
    if not out:
        raise ValueError("All rows were malformed.")
    if skipped > 0:
        logger.warning("Skipped %d malformed rows", skipped)
    return out


def build_prompt(question: str, choices: List[str]) -> str:
    prompt = f"Question: {question}\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}. {choice}\n"
    prompt += "Answer:"
    return prompt


def score_choice_text_batch(
    model,
    tokenizer,
    device: str,
    prompt_choice_pairs: List[tuple[str, str]],
    batch_size: int,
) -> List[tuple[float, float, int]]:
    encoded = []
    for prompt, choice in prompt_choice_pairs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        choice_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
        if not choice_ids:
            raise ValueError("Choice tokenized to zero tokens")
        encoded.append((prompt_ids, choice_ids, prompt_ids + choice_ids))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs a pad token for batched scoring.")

    scores: List[tuple[float, float, int]] = []
    for start_idx in range(0, len(encoded), batch_size):
        batch = encoded[start_idx : start_idx + batch_size]
        max_len = max(len(input_ids) for _, _, input_ids in batch)
        input_ids = []
        attention_mask = []
        for _, _, ids in batch:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits

        for batch_idx, (prompt_ids, choice_ids, _) in enumerate(batch):
            score_start = len(prompt_ids) - 1
            score_end = score_start + len(choice_ids)
            choice_logits = logits[batch_idx, score_start:score_end, :]
            labels = torch.tensor(choice_ids, dtype=torch.long, device=device)
            token_nll = F.cross_entropy(choice_logits, labels, reduction="none")
            nll_sum = float(token_nll.sum().item())
            mean_nll = nll_sum / len(choice_ids)
            scores.append((-mean_nll, nll_sum, len(choice_ids)))
    return scores


def mean_or_zero(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        default="./banks",
        help="Path to bank file or directory (supports .jsonl/.json).",
    )
    parser.add_argument(
        "--banks",
        nargs="+",
        default=None,
        help="Optional list of bank files/directories to load together. If set, overrides --bank.",
    )
    parser.add_argument(
        "--mmlu",
        action="store_true",
        help="Evaluate all MMLU banks under data/targets/mmlu_banks except auxiliary_train.",
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
        default="bank_eval_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on number of questions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of answer-choice continuations per forward pass.",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--save-by-question",
        action="store_true",
        help="Include per-question predictions and losses in output JSON.",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.mmlu:
        mmlu_dir = Path(__file__).resolve().parent.parent / "data" / "targets" / "mmlu_banks"
        bank_paths = sorted(
            p
            for p in mmlu_dir.glob("*.jsonl")
            if p.name != "auxiliary_train.jsonl"
        )
        logger.info("Loading rows from %d MMLU bank paths (--mmlu)", len(bank_paths))
    else:
        bank_paths = [Path(p) for p in args.banks] if args.banks else [Path(args.bank)]
    if args.banks and not args.mmlu:
        logger.info("Loading rows from %d bank paths (--banks)", len(bank_paths))
    elif not args.mmlu:
        logger.info("Loading bank rows from %s", bank_paths[0])

    all_rows: List[dict] = []
    for bank_path in bank_paths:
        logger.info("Loading bank rows from %s", bank_path)
        all_rows.extend(load_bank(bank_path))
    rows = normalize_rows(all_rows)
    if args.max_questions is not None:
        rows = rows[: args.max_questions]
        logger.info("Applied --max-questions=%d (using %d rows)", args.max_questions, len(rows))
    logger.info("Evaluating %d questions", len(rows))

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device == "auto":
        device = "cpu"
    dtype = parse_dtype(args.dtype)
    logger.info("Using device=%s dtype=%s", device, args.dtype)

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

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    total_loss = 0.0
    total_correct = 0
    by_dataset: Dict[str, dict] = defaultdict(lambda: {"num_questions": 0, "num_correct": 0, "loss_sum": 0.0})
    by_question: List[dict] = []
    processed_questions = 0

    def process_scored_rows(pending_rows: List[dict]) -> None:
        nonlocal total_loss, total_correct, processed_questions
        prompt_choice_pairs: List[tuple[str, str]] = []
        choice_counts: List[int] = []
        for row in pending_rows:
            prompt = build_prompt(row["question"], row["choices"])
            choice_counts.append(len(row["choices"]))
            prompt_choice_pairs.extend((prompt, choice) for choice in row["choices"])

        flat_scores = score_choice_text_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_choice_pairs=prompt_choice_pairs,
            batch_size=args.batch_size,
        )

        offset = 0
        for row, num_choices in zip(pending_rows, choice_counts):
            dataset = row["dataset"]
            answer_idx = row["answer_idx"]
            scored_choices = flat_scores[offset : offset + num_choices]
            offset += num_choices

            option_scores = torch.tensor([x[0] for x in scored_choices], dtype=torch.float32, device=device)
            loss = F.cross_entropy(option_scores.unsqueeze(0), torch.tensor([answer_idx], device=device))
            pred_idx = int(torch.argmax(option_scores).item())
            pred_prob = float(torch.softmax(option_scores, dim=-1)[pred_idx].item())

            correct = int(pred_idx == answer_idx)
            loss_value = float(loss.item())

            total_correct += correct
            total_loss += loss_value
            by_dataset[dataset]["num_questions"] += 1
            by_dataset[dataset]["num_correct"] += correct
            by_dataset[dataset]["loss_sum"] += loss_value
            processed_questions += 1

            if args.save_by_question:
                by_question.append(
                    {
                        "id": row["id"],
                        "dataset": dataset,
                        "answer_idx": answer_idx,
                        "pred_idx": pred_idx,
                        "correct": bool(correct),
                        "loss": loss_value,
                        "pred_prob": pred_prob,
                        "choice_mean_nlls": [-x[0] for x in scored_choices],
                        "choice_token_counts": [x[2] for x in scored_choices],
                    }
                )

            if processed_questions % 50 == 0:
                logger.info(
                    "Progress: %d/%d | accuracy=%.4f mean_loss=%.4f",
                    processed_questions,
                    len(rows),
                    mean_or_zero(total_correct, processed_questions),
                    mean_or_zero(total_loss, processed_questions),
                )

    pending_rows: List[dict] = []
    pending_choices = 0
    for row in rows:
        pending_rows.append(row)
        pending_choices += len(row["choices"])
        if pending_choices >= args.batch_size:
            process_scored_rows(pending_rows)
            pending_rows = []
            pending_choices = 0
    if pending_rows:
        process_scored_rows(pending_rows)

    per_dataset = {}
    for dataset, stats in sorted(by_dataset.items()):
        n = int(stats["num_questions"])
        n_correct = int(stats["num_correct"])
        loss_sum = float(stats["loss_sum"])
        per_dataset[dataset] = {
            "num_questions": n,
            "num_correct": n_correct,
            "accuracy": mean_or_zero(n_correct, n),
            "mean_loss": mean_or_zero(loss_sum, n),
        }

    num_questions = len(rows)
    output = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bank": str(bank_paths[0]) if len(bank_paths) == 1 else None,
            "banks": [str(p) for p in bank_paths],
            "mmlu": args.mmlu,
            "model": args.model,
            "tokenizer": tokenizer_name,
            "device": device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "num_questions": num_questions,
            "save_by_question": args.save_by_question,
            "scoring": "normalized_answer_text",
        },
        "overall": {
            "num_questions": num_questions,
            "num_correct": total_correct,
            "accuracy": mean_or_zero(total_correct, num_questions),
            "mean_loss": mean_or_zero(total_loss, num_questions),
        },
        "by_dataset": per_dataset,
    }
    if args.save_by_question:
        output["by_question"] = by_question

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation JSON to %s", out_path)
    logger.info(
        "Overall: accuracy=%.4f mean_loss=%.4f (%d questions)",
        output["overall"]["accuracy"],
        output["overall"]["mean_loss"],
        num_questions,
    )


if __name__ == "__main__":
    main()
