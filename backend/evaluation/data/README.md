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

## Additional sets (added after the first round of tuning)

| File | Items | Class | Purpose |
|---|---|---|---|
| `heldout_adversarial.json` | 40 | attack | Written after a first round of detector work on the first 60, to see how they generalise to new wording (see the caveat below). |
| `technical_benign.json` | 40 | benign | Ordinary engineering facts and preferences (versions, ports, tooling), the register where false positives are most likely. |
| `legitimate_updates.json` | 20 | benign | Genuine corrections ("I moved to the afternoon shift") that look like revisions but are not attacks. |
| `evasion_adversarial.json` | 30 | attack | Paraphrased versions of attacks that avoid the keywords and sentence shapes the detectors look for. Scored once, after everything else was frozen. |

All four sets are written by the project team for this repository. They are not
externally reviewed. The held-out set is not clean: after it was scored once with
the original detectors, the misses were read, and they shaped the new detector
families (third-person authority claims, passive-voice policy changes, conditional
directives). Its recall of 82.5% is therefore optimistic. The evasion set was
written after the detectors were frozen and scored once, and it is the more honest
test of generalisation (6.7% recall; see the root README).

## Benchmark v4: public sources

The 120 authored cases above are too few and too close to the people who wrote the detectors, so the benchmark
now also draws on public data. The public files are not committed. `python -m evaluation.public_data` downloads
the upstream files once into `public_cache/` and writes the selected items to `public/` (both git-ignored). Every
random choice uses a fixed seed, so two machines build the same sets.

| Source | Kind | Items | What it is | License |
|---|---|---|---|---|
| `injecagent` | attack | 62 | Attacker instructions from InjecAgent (tool-use attacks) | see repository |
| `deepset` | attack | 154 | English injections in `deepset/prompt-injections` | Apache-2.0 |
| `gandalf` | attack | 150 | `Lakera/gandalf_ignore_instructions` | MIT |
| `safeguard` | attack | 300 | Injections in `xTRam1/safe-guard-prompt-injection` | unspecified |
| `poisonedrag` | attack | 200 | Poisoned passages from PoisonedRAG, with the question each one targets | MIT |
| `personachat` | benign | 392 | PersonaChat persona statements, tidied | unspecified |
| `oasst1` | benign | 228 | English OpenAssistant user messages that state preferences, setups or standing instructions | Apache-2.0 |

Together with the authored sets that is 1,736 cases (996 attacks, 740 benign). Items are de-duplicated across
sources, capped at 400 characters, and split into train (60%) and test (40%) by a hash of the text. The authored
development set is train; the authored held-out and paraphrase sets are test.

Things to know before reading any number from this benchmark:

- **Attack labels are about content, not provenance.** These are prompts and passages that someone wrote to
  manipulate an assistant. Whether storing one is harmful in a real deployment depends on where it came from.
  A benign user preference ("always answer in French") and an injected instruction can read alike.
- **PoisonedRAG items are false facts.** A text classifier cannot know they are false. They are scored as attacks
  anyway, and they are the one place where historical consistency against a stored fact matters (see
  `history_eval.py`).
- **Style differs between sources.** PersonaChat statements are short and informal; the attacks are longer.
  A classifier that only learned style would look good here. A style-only control (length, punctuation,
  capitalisation) is run alongside, and leave-one-source-out training tests whether the classifier carries over to
  a source it has not seen. OpenAssistant is included for exactly this reason: its benign messages are long and
  instruction-heavy.
- **Nothing public was inspected before the rules were frozen.** The rule-based prototype was scored on every item
  before any learned component was built, and that run is kept in `results/frozen_heuristics_v4.json`.

`profile_contradictions.json` (259 cases, committed) is generated: each case has three PersonaChat statements
that count as accepted memories, one consistent new statement, and one statement that contradicts a stored line.
The contradictions were written by a language model from the fixed prompt stored in the file.

## Evaluation protocols

The first version of the benchmark built each candidate memory from the ground
truth: attack items were sent as `attack_simulator` source with category
`instruction`, benign items as `user_conversation` with category `fact`. That
hands the defence the answer through two of the six signals. The runner now
supports four protocols, and the headline numbers use the first one:

- `pipeline_static`: every item enters as a user message, and source and category
  come from the same extractor the live system uses. The store is the fixed
  academic baseline. This is the primary protocol.
- `pipeline_streaming`: same construction, but items arrive one at a time, accepted
  memories are written to the store, and the isolation forest refits as the store
  grows.
- `pipeline_unauth`: as static, but attack items arrive on an untrusted channel.
  This is the setting a provenance gate is designed for.
- `label_informed_legacy`: the original construction, kept only as an ablation to
  show how much the label leak inflated the result.

## Reproducibility

All files are static JSON, loaded by `evaluation/datasets.py`. The primary
benchmark is the 60 PersonaChat items plus the 60 development attacks, shuffled
with seed 42. Runs need no network access (`GEMINI_API_KEY` empty uses the local
hash embedder) and are deterministic for a given seed.

```bash
cd backend
python -m evaluation.run_report --seeds 10    # writes evaluation/results/latest.json
```
