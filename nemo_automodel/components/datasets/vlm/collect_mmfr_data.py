"""Self-contained MMFR dataset loader, shaped like nemo_automodel's make_cord_v2_dataset.

The function name `make_cord_v2_dataset` is preserved per the project's YAML config;
it actually loads the MMFR fake/real subset.

Pipeline:
    1. Download shards from AnnaGao/MMFR-Dataset on demand via huggingface_hub
       (cached under HF_HOME). Always download dataset.part1.tar first because it
       contains forgery_reasoning_cot.json (the per-image CoT index).
    2. For each ordered shard (FAKE_SHARDS / REAL_SHARDS), list the images present
       and pull bytes for entries the CoT JSON has reasoning for. Stop once we
       have enough rows.
    3. Format each row into a {"conversation": [user, assistant]} sample.

Output format the assistant is trained to produce:

    <think>
    {per-image reasoning, pulled from MMFR's forgery_reasoning_cot.json}
    </think>

    ```json
    {"status": "ai_generated"|"real", "confidence": 1.0, "reason": "..."}
    ```
"""

import io
import json
import re
import subprocess

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "AnnaGao/MMFR-Dataset"
COT_JSON_TARPATH = "dataset/forgery_reasoning_cot.json"

# Shard ranges discovered empirically:
#   parts 1-69  -> diffusiondb (fake)         — part1 also carries the CoT JSON
#   parts 70-79 -> evaluation_sets (skip)
#   parts 80-86 -> laion (real)
FAKE_SHARDS = [f"dataset.part{i}.tar" for i in range(1, 70)]
REAL_SHARDS = [f"dataset.part{i}.tar" for i in range(80, 87)]

FAKE_JSON_PREFIX = "diffusiondb/"
REAL_JSON_PREFIX = "laion/"

USER_PROMPT = (
    "Analyze the provided image and determine whether it is AI-generated.\n"
    "Think through the visual cues, then respond with a JSON object with these fields:\n"
    "  - status: either \"ai_generated\" or \"real\"\n"
    "  - confidence: a number between 0 and 1\n"
    "  - reason: a brief explanation of your decision"
)

_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.DOTALL)


def _extract_member(tar_path: str, member: str) -> bytes | None:
    """Stream a single tar member's bytes via system tar (tolerant of MMFR's tar prefix)."""
    proc = subprocess.run(
        ["tar", "-xOf", tar_path, member],
        capture_output=True, check=False,
    )
    return proc.stdout or None


def _list_images_in_shard(tar_path: str, json_prefix: str) -> set[str]:
    """Return the set of image paths (CoT-JSON-style, no 'dataset/' prefix) in tar_path."""
    proc = subprocess.run(
        ["tar", "-tf", tar_path],
        capture_output=True, text=True, check=False,
    )
    members = set()
    for name in proc.stdout.splitlines():
        if not name.startswith("dataset/" + json_prefix):
            continue
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        members.add(name[len("dataset/"):])
    return members


def _short_reason(reasoning: str, max_chars: int = 240) -> str:
    text = reasoning.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    return (cut[: last_period + 1] if last_period > 50 else cut).strip()


def _gather_across_shards(cot_filtered, shard_list, json_prefix, total_needed, label):
    """Walk shards in order, extracting CoT-indexed images, until total_needed is hit."""
    rows = []
    consumed: set[str] = set()
    for shard in shard_list:
        if len(rows) >= total_needed:
            break
        try:
            tar_path = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset")
        except Exception as e:
            print(f"[warn] couldn't download {shard}: {e}")
            continue
        in_shard = _list_images_in_shard(tar_path, json_prefix)
        added = 0
        for entry in cot_filtered:
            if len(rows) >= total_needed:
                break
            img_rel = entry["image"]
            if img_rel in consumed or img_rel not in in_shard:
                continue
            data = _extract_member(tar_path, "dataset/" + img_rel)
            consumed.add(img_rel)
            if not data:
                continue
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue
            gpt_resp = entry["conversations"][1]["value"]
            m = _REASONING_RE.search(gpt_resp)
            reasoning = m.group(1).strip() if m else gpt_resp.strip()
            rows.append({"image": img, "label": label, "reasoning": reasoning})
            added += 1
        print(f"[shard {shard}] +{added} {('fake' if label == 1 else 'real')} rows  (total {len(rows)})")
    return rows


def make_cord_v2_dataset(
    path_or_dataset=REPO_ID,
    split="train",
    n_train_per_class=100,
    n_val_per_class=20,
    **kwargs,
):
    """Load and preprocess the MMFR fake/real subset for image+text VLM fine-tuning.

    Train and validation slices are disjoint: train takes the first
    ``n_train_per_class`` rows of each class; validation takes the next
    ``n_val_per_class``. Shards are downloaded on demand until enough rows
    are collected (cap depends on dataset size — currently up to ~50k per class).
    """
    # part1.tar always needed for the CoT JSON
    part1 = hf_hub_download(repo_id=path_or_dataset, filename="dataset.part1.tar", repo_type="dataset")
    cot_blob = _extract_member(part1, COT_JSON_TARPATH)
    if not cot_blob:
        raise RuntimeError(f"Could not extract {COT_JSON_TARPATH} from {part1}")
    cot = json.loads(cot_blob)

    cot_fake = [e for e in cot if e["image"].startswith(FAKE_JSON_PREFIX)]
    cot_real = [e for e in cot if e["image"].startswith(REAL_JSON_PREFIX)]
    print(f"[cot] {len(cot_fake)} fake / {len(cot_real)} real candidate entries in {split} split")

    total = n_train_per_class + n_val_per_class
    fake_rows = _gather_across_shards(cot_fake, FAKE_SHARDS, FAKE_JSON_PREFIX, total, label=1)
    real_rows = _gather_across_shards(cot_real, REAL_SHARDS, REAL_JSON_PREFIX, total, label=0)

    '''
    if split == "train":
        dataset = fake_rows[:n_train_per_class] + real_rows[:n_train_per_class]
    elif split in ("validation", "val", "test"):
        dataset = fake_rows[n_train_per_class:] + real_rows[n_train_per_class:]
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")
    '''
    if split == "train":
        dataset = fake_rows[n_val_per_class:] + real_rows[n_val_per_class:]
    elif split in ("validation", "val", "test"):
        dataset = fake_rows[:n_val_per_class] + real_rows[:n_val_per_class]
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def format(example):
        obj = {
            "status": "ai_generated" if example["label"] == 1 else "real",
            "confidence": 1.0,
            "reason": _short_reason(example["reasoning"]),
        }
        assistant_text = (
            f"<think>\n{example['reasoning'].strip()}\n</think>\n\n"
            f"```json\n{json.dumps(obj, indent=2)}\n```"
        )
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ],
        }

    return [format(example) for example in dataset]