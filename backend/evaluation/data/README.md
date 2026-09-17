# ATLAS Benchmark: Dataset Provenance

This benchmark is deliberately grounded in external sources rather than assembled
from scratch: the benign class is drawn from a real, peer-reviewed public dataset
rather than written by the project team, and the adversarial class is organized
under a published attack-risk taxonomy rather than assembled ad hoc.

## `benign_seed.json` (60 items)

**Source:** [PersonaChat](https://arxiv.org/abs/1801.07243) (Zhang, Dinan, Urbanek,
Szlam, Kiela, Weston, "Personalizing Dialogue Agents: I have a dog, do you have
pets too?", ACL 2018), a widely-cited, peer-reviewed crowd-sourced dialogue
dataset. Statements were pulled from each speaker's `personality` field (a set of
first-person facts/preferences the crowd-worker was asked to embody), via the
`AlekseyKorshuk/persona-chat` mirror on the Hugging Face Hub.

**Construction pipeline** (reproducible; see `git log` / this file for the exact steps):
1. Loaded the full `train` split (~18k conversations), flattened and deduplicated
   all `personality` lines (5,787 unique statements).
2. Filtered for length (15-90 characters, at least 3 words) and removed lines
   containing sensitive topics (medical conditions, self-harm, substance abuse,
   etc.; see the `sensitive_kw` exclusion list used during construction) since
   these are inappropriate for a public academic demo regardless of dataset origin.
3. Randomly sampled with a **fixed seed (42)** for full reproducibility, then
   manually spot-checked for grammatical coherence.
4. Tagged each surviving statement as `"preference"` or `"fact"` using simple
   marker-word heuristics, and capitalized the first letter.

**Known limitation:** PersonaChat statements are casual/personal in register
("i like to remodel homes", "my favorite pizza is pepperoni") rather than
software/infrastructure statements. This is a deliberate trade-off: it trades
topical narrowness for genuine, independently-verifiable external provenance.
If the benign class needs to be restricted to a technical/software domain
specifically, PersonaChat is not a perfect match, and a domain-specific
alternative (e.g. a developer-forum or support-ticket corpus) would need to be
sourced separately.

## `adversarial_seed.json` (60 items)

**No existing public benchmark could be directly reused** for this class as of
September 2026:

- **WhisperBench** (introduced in MemGhost, Zhang et al. 2026, arXiv:2607.05189)
  is the closest match in spirit: 108 cases across exactly this kind of
  fact/preference poisoning, but its authors are holding the dataset back
  under responsible disclosure until affected agent frameworks are patched,
  so it is not yet publicly downloadable.
- **MINJA**'s official repository (github.com/dsh3n77/MINJA, NeurIPS 2025) does
  not include a released, directly reusable case file in its README.

Given that, this file's 60 cases were **constructed by the project team**,
but systematically rather than ad hoc:

- Organized into the **same five risk categories** WhisperBench introduces
  (`health_and_safety`, `financial_loss`, `information_integrity`,
  `cybersecurity`, `operational_disruption`), 12 cases each, so the benchmark
  is at least taxonomically aligned with published work even though the
  specific case text is our own.
- Each case is tagged with an `attack_technique` field (`direct_fact_overwrite`,
  `covert_instruction_injection`, `semantic_bridging`, `entity_masquerading`,
  `code_safety_evasion`, `sybil_authority_spoof`) and an `inspired_by` field
  citing the specific paper whose described mechanism the phrasing follows
  (MINJA's bridging/future-conditioning style, MemPoison's semantic-bridging
  and entity-masquerading style, Sunil et al.'s code/tool-safety heuristics).

**Known limitation, stated plainly:** this class is still team-authored text,
not an externally reviewed benchmark. It is a *taxonomically grounded*
construction, not a substitute for the real thing. Once WhisperBench or an
equivalent MINJA/MemPoison case file is publicly released, swapping it in here
is the natural next step.

## Reproducibility

Both files are static JSON, loaded once at import time by
`evaluation/experiment_runner.py::_load_benchmark_dataset()`, then shuffled
with a fixed seed (42) before evaluation. The benchmark requires no network
access at run time and produces identical results on every run.
