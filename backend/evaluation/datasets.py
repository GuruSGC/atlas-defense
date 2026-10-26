"""
Benchmark data loading.

Sets
  dev_adversarial       60 adversarial cases used while building and tuning the detectors
  heldout_adversarial   40 adversarial cases written after a first round of detector work (optimistic, see data README)
  benign_personachat    60 casual statements sampled from PersonaChat
  benign_technical      40 developer-style statements, 19 of them hard negatives that use
                        words such as "always", "never", "approved", or "delete" innocently
  legitimate_updates    20 honest user updates that use revision language
  evasion_adversarial   paraphrased attacks written after the detectors were frozen
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_SEED = 42


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    context: str
    is_attack: bool
    set_name: str
    technique: str = ""
    category: str = ""
    source: str = ""
    split: str = ""
    probe: str = ""          # a later question that should retrieve this memory (PoisonedRAG items)
    incorrect: str = ""      # the wrong answer the poisoned memory pushes (PoisonedRAG items)
    correct: str = ""


def _read(name: str) -> list:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _adversarial(name: str, set_name: str) -> List[Item]:
    path = DATA_DIR / name
    if not path.exists():
        return []
    return [Item(r["id"], r["text"], r["context"], True, set_name, r.get("attack_technique", ""), r.get("category", ""))
            for r in _read(name)]


def _benign(name: str, set_name: str) -> List[Item]:
    return [Item(r["id"], r["text"], r["text"], False, set_name, "", r.get("category", ""))
            for r in _read(name)]


def load_sets() -> Dict[str, List[Item]]:
    return {
        "dev_adversarial": _adversarial("adversarial_seed.json", "dev_adversarial"),
        "heldout_adversarial": _adversarial("heldout_adversarial.json", "heldout_adversarial"),
        "benign_personachat": _benign("benign_seed.json", "benign_personachat"),
        "benign_technical": _benign("technical_benign.json", "benign_technical"),
        "legitimate_updates": _benign("legitimate_updates.json", "legitimate_updates"),
        "evasion_adversarial": _adversarial("evasion_adversarial.json", "evasion_adversarial"),
    }


def primary_benchmark(seed: int = DATASET_SEED) -> List[Item]:
    """The 120-item benchmark (60 PersonaChat + 60 dev adversarial), shuffled with a fixed seed."""
    sets = load_sets()
    items = list(sets["benign_personachat"]) + list(sets["dev_adversarial"])
    random.Random(seed).shuffle(items)
    return items


# ---- benchmark v4: authored sets plus public sources --------------------------------------------------

import hashlib

PUBLIC_DIR = DATA_DIR / "public"
PUBLIC_SOURCES = ["injecagent", "deepset", "gandalf", "safeguard", "poisonedrag", "personachat", "oasst1"]
_AUTHORED_SPLIT = {"dev_adversarial": "train", "heldout_adversarial": "test", "evasion_adversarial": "test"}


def _hash_split(text: str) -> str:
    return "test" if int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 100 < 40 else "train"


def public_available() -> bool:
    return all((PUBLIC_DIR / f"{n}.jsonl").exists() for n in PUBLIC_SOURCES)


def _public(name: str) -> List[Item]:
    path = PUBLIC_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(Item(r["id"], r["text"], r["context"], r["is_attack"], name, r.get("category", ""), r.get("category", ""),
                              name, r["split"], r.get("probe", ""), r.get("incorrect", ""), r.get("correct", "")))
    return items


def load_benchmark(include_public: bool = True) -> Dict[str, List[Item]]:
    """Every source of the v4 benchmark, keyed by source name. Authored sources are always present; the
    public ones appear when evaluation.public_data has been run."""
    out: Dict[str, List[Item]] = {}
    seen = set()
    for name, items in load_sets().items():
        rows = []
        for it in items:
            split = _AUTHORED_SPLIT.get(name) or _hash_split(it.text)
            rows.append(Item(it.id, it.text, it.context, it.is_attack, name, it.technique, it.category, "authored:" + name, split))
            seen.add(it.text.lower())
        out["authored:" + name] = rows
    if include_public:
        for name in PUBLIC_SOURCES:
            rows = [it for it in _public(name) if it.text.lower() not in seen]
            seen.update(it.text.lower() for it in rows)
            if rows:
                out[name] = rows
    return out


def benchmark_items(split: str = None, include_public: bool = True) -> List[Item]:
    items = [it for rows in load_benchmark(include_public).values() for it in rows]
    if split:
        items = [it for it in items if it.split == split]
    return items
