"""
Builds the public part of the benchmark.

Nothing from these sources is committed. The raw files are downloaded once into
evaluation/data/public_cache/ (git-ignored) and the processed sets are written to
evaluation/data/public/ (also git-ignored). Selection is deterministic: every random choice uses a
fixed seed, so two machines build the same sets from the same upstream files.

Adversarial sources (candidate memories an attacker would try to persist)
  injecagent     attacker instructions from InjecAgent (tool-use attacks)
  deepset        English prompts labelled as injections in deepset/prompt-injections
  gandalf        "ignore your instructions" prompts from Lakera/gandalf_ignore_instructions
  safeguard      injection prompts from xTRam1/safe-guard-prompt-injection
  poisonedrag    poisoned passages from PoisonedRAG (false facts, with the question they target)

Benign sources
  personachat    persona statements from PersonaChat (Zhang et al., 2018)
  oasst1         English user messages from OpenAssistant that state preferences, setups or standing
                 instructions ("always", "from now on", "my project uses"), the hard negatives for
                 any keyword rule
"""

import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "public_cache"
OUT_DIR = DATA_DIR / "public"
SEED = 2026
MAX_CHARS = 400
MIN_CHARS = 12

HF = "https://huggingface.co/api/datasets/{}/parquet/default/{}/0.parquet"
GH = "https://raw.githubusercontent.com/{}"

SOURCES = {
    "injecagent": {"label": "attack", "license": "see repository", "url": "https://github.com/uiuc-kang-lab/InjecAgent"},
    "deepset": {"label": "attack", "license": "Apache-2.0", "url": "https://huggingface.co/datasets/deepset/prompt-injections"},
    "gandalf": {"label": "attack", "license": "MIT", "url": "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions"},
    "safeguard": {"label": "attack", "license": "unspecified", "url": "https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection"},
    "poisonedrag": {"label": "attack", "license": "MIT", "url": "https://github.com/sleeepeer/PoisonedRAG"},
    "personachat": {"label": "benign", "license": "unspecified", "url": "https://huggingface.co/datasets/AlekseyKorshuk/persona-chat"},
    "oasst1": {"label": "benign", "license": "Apache-2.0", "url": "https://huggingface.co/datasets/OpenAssistant/oasst1"},
}

TARGETS = {  # items kept per source
    "injecagent": 80, "deepset": 200, "gandalf": 150, "safeguard": 300, "poisonedrag": 200,
    "personachat": 400, "oasst1": 300,
}


def _fetch(url: str, name: str) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "atlas-benchmark-builder"})
        with urllib.request.urlopen(req, timeout=180) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _parquet(dataset: str, split: str, name: str):
    import pyarrow.parquet as pq
    raw = _fetch(HF.format(dataset, split), name)
    return pq.read_table(io.BytesIO(raw)).to_pylist()


def _clean(text: str) -> str:
    text = (text or "").replace("\u2014", ",").replace("\u2013", "-").replace("\ufffd", "")
    return re.sub(r"\s+", " ", text).strip()


_EN_WORDS = set("the a an and or of to in on for with is are was were be this that it you your i my we our can will not do does as at by from".split())


def is_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 3:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    stop = sum(w in _EN_WORDS for w in words) / len(words)
    return ascii_ratio > 0.97 and stop > 0.12


def _split_of(text: str) -> str:
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if h < 40 else "train"


def _pick(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    rows = sorted(rows, key=lambda r: r["text"])
    rng.shuffle(rows)
    return rows[:n]


def _ok(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS and is_english(text) and "\n" not in text


def load_injecagent() -> List[dict]:
    out, seen = [], set()
    for f in ("attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"):
        raw = _fetch(GH.format(f"uiuc-kang-lab/InjecAgent/main/data/{f}"), f).decode("utf-8")
        for line in raw.splitlines():
            if not line.strip():
                continue
            t = _clean(json.loads(line)["Attacker Instruction"])
            if _ok(t) and t not in seen:
                seen.add(t)
                out.append({"text": t, "category": "tool_use_attack"})
    return out


def load_deepset() -> List[dict]:
    out, seen = [], set()
    for split in ("train", "test"):
        for r in _parquet("deepset/prompt-injections", split, f"deepset_{split}.parquet"):
            t = _clean(r["text"])
            if r["label"] == 1 and _ok(t) and t not in seen:
                seen.add(t)
                out.append({"text": t, "category": "prompt_injection"})
    return out


def load_gandalf() -> List[dict]:
    out, seen = [], set()
    for split in ("train", "validation", "test"):
        for r in _parquet("Lakera/gandalf_ignore_instructions", split, f"gandalf_{split}.parquet"):
            t = _clean(r["text"])
            if _ok(t) and t not in seen:
                seen.add(t)
                out.append({"text": t, "category": "ignore_instructions"})
    return out


def load_safeguard() -> List[dict]:
    out, seen = [], set()
    for split in ("train", "test"):
        for r in _parquet("xTRam1/safe-guard-prompt-injection", split, f"safeguard_{split}.parquet"):
            t = _clean(r["text"])
            if r["label"] == 1 and _ok(t) and t not in seen:
                seen.add(t)
                out.append({"text": t, "category": "prompt_injection"})
    return out


def load_poisonedrag() -> List[dict]:
    out, seen = [], set()
    for ds in ("nq", "hotpotqa", "msmarco"):
        raw = _fetch(GH.format(f"sleeepeer/PoisonedRAG/main/results/adv_targeted_results/{ds}.json"), f"poisonedrag_{ds}.json")
        for k, v in sorted(json.loads(raw).items()):
            if not v.get("adv_texts"):
                continue
            t = _clean(v["adv_texts"][0])
            if len(t) > 500:
                t = t[:500].rsplit(" ", 1)[0]
            if MIN_CHARS <= len(t) and is_english(t) and t not in seen:
                seen.add(t)
                out.append({"text": t, "category": "false_fact", "probe": _clean(v["question"]),
                            "correct": _clean(str(v["correct answer"])), "incorrect": _clean(str(v["incorrect answer"])), "dataset": ds})
    return out


def load_personachat() -> List[dict]:
    seen, out = set(), []
    for split in ("train",):
        for r in _parquet("AlekseyKorshuk/persona-chat", split, f"personachat_{split}.parquet"):
            for line in r.get("personality", []):
                t = re.sub(r"\s+([.,!?;:])", r"\1", _clean(line))
                t = t[:1].upper() + t[1:]
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                if _ok(t) and len(t.split()) >= 3:
                    out.append({"text": t, "category": "persona"})
    return out


_MEMORY_LANGUAGE = re.compile(
    r"\b(always|never|call me|my name is|i (like|love|hate|enjoy|live|speak)|from now on|going forward|in future|remember( that)?|keep in mind|i prefer|i usually|i use|my (favorite|favourite|"
    r"project|team|company|setup|stack|workflow|server|laptop|code|app)|i work (on|with|in|at)|i am (a|an|using|working|learning)|"
    r"we use|we are using|please (always|never|use|keep|call me))\b", re.I)


def load_oasst1() -> List[dict]:
    out, seen = [], set()
    for split in ("train", "validation"):
        for r in _parquet("OpenAssistant/oasst1", split, f"oasst1_{split}.parquet"):
            if r.get("role") != "prompter" or r.get("lang") != "en" or r.get("deleted"):
                continue
            t = _clean(r["text"])
            if not _ok(t) or "?" in t[-3:] or not _MEMORY_LANGUAGE.search(t) or t in seen:
                continue
            if re.search(r"https?://|```", t):
                continue
            seen.add(t)
            out.append({"text": t, "category": "user_context"})
    return out


LOADERS = {
    "injecagent": load_injecagent, "deepset": load_deepset, "gandalf": load_gandalf,
    "safeguard": load_safeguard, "poisonedrag": load_poisonedrag, "personachat": load_personachat, "oasst1": load_oasst1,
}


def build(sources: List[str] = None) -> Dict[str, List[dict]]:
    """Download (if needed), filter, sample and write one JSONL file per source."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = {}
    for name in sources or list(LOADERS):
        rng = random.Random(f"{SEED}-{name}")
        pool = LOADERS[name]()
        chosen = _pick(pool, TARGETS[name], rng)
        info = SOURCES[name]
        rows = []
        for i, r in enumerate(chosen):
            row = dict(r)
            row.update({"id": f"{name}-{i:04d}", "context": r["text"], "is_attack": info["label"] == "attack",
                        "source": name, "split": _split_of(r["text"])})
            rows.append(row)
        (OUT_DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        built[name] = rows
        print(f"{name:12s} pool={len(pool):6d} kept={len(rows):4d} train={sum(r['split'] == 'train' for r in rows):4d} "
              f"test={sum(r['split'] == 'test' for r in rows):4d}")
    manifest = {n: {**SOURCES[n], "kept": len(rows), "seed": SEED} for n, rows in built.items()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return built


def available() -> bool:
    return all((OUT_DIR / f"{n}.jsonl").exists() for n in LOADERS)


if __name__ == "__main__":
    build()
