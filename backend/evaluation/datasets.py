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
