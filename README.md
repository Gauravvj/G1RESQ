# G1RESQ

**Offline-first multimodal AI disaster response and emergency coordination system.**

G1RESQ helps citizens report emergencies (text first, then voice and photos), uses a locally running AI model to assess how urgent a report is, and helps responders and coordinators prioritise. The core design goal is that essential functionality keeps working **without internet access**.

> **Important:** AI output is decision support only. It is not a medical diagnosis and never makes dispatch decisions. A human reviews every triage and dispatch decision.

---

## Current status

We are in **Phase 1: the AI engine** (text-based emergency assessment with Gemma 3n). Backend, frontend and offline sync come later.

| Milestone | What | Status |
|---|---|---|
| M0 | Run Gemma 3n E2B locally through Ollama, measure speed and RAM | Done |
| M1 | Schema, versioned prompt, JSON validation, retry, safe fallback | Done |
| M2 | Rule-based safety layer (urgency floor) and speed pass | **In progress** |
| M3 | Gold test set and evaluation harness | Planned |
| M4 | Package as one clean function, `assess_report(text)`, ready for FastAPI | Planned |

Reference numbers (8 GB RAM laptop, Intel i5-1240P, CPU only, Gemma 3n E2B): about **14 s per report**, about **6.4 GB** of RAM used by the model. These come from 10 hand-picked reports, not a benchmark. The real measurements arrive with M3.

---

## How the AI engine works

```
report text
   |
   v
Safety rules (keywords, < 1 ms) ---> urgency floor + red flags
   |
   v
Prompt (versioned) --> Gemma 3n E2B via Ollama (JSON-schema constrained output)
   |
   v
Validate with Pydantic --> on failure, retry once with the error fed back --> else safe fallback
   |
   v
Evidence check (do the quoted phrases really appear in the report?)
   |
   v
Final urgency = max(LLM urgency, rule floor)      Summary and missing info are computed in code
   |
   v
IncidentAssessment (validated result, always flagged for human review when it matters)
```

Design principles we follow:

1. **The model is replaceable.** All inference sits behind one interface (`backends/base.py`).
2. **The LLM never has the last word on safety.** Rules can only *raise* urgency, never lower it.
3. **Every output is validated.** If the model fails twice, we return a safe fallback (high urgency, human review), never a crash or a guess.
4. **Compute in code what code can compute.** Missing information and the one-line summary are built from validated fields, not written by the LLM.
5. **Measure everything.** No prompt change or model swap is accepted without numbers on the same test set.

---

## Repository layout

```
g1resq/
├── ai_engine/                     Phase 1: importable Python package (no web dependencies)
│   ├── g1resq_ai/
│   │   ├── backends/              base.py (interface), ollama_backend.py
│   │   ├── prompts/               triage_v1.py, triage_v2.py, triage_v3.py (one file per version)
│   │   ├── safety/rules.py        rule-based urgency floor (English, Hinglish, Hindi)
│   │   ├── config.py              settings and defaults
│   │   ├── schemas.py             Pydantic models: LLMAssessment, IncidentAssessment
│   │   ├── parsing.py             JSON extraction, validation, evidence check
│   │   └── pipeline.py            assess_report(): the single entry point
│   ├── evaluation/                gold.py (test-set format and validator); run_eval.py is a placeholder
│   ├── data/gold/gold_v0.jsonl    hand-labelled test reports
│   ├── scripts/                   m0_benchmark.py, m1_try_reports.py, m2_speed_diagnosis.py
│   ├── results/                   script outputs (git-ignored)
│   ├── tests/                     pytest tests (no Ollama needed)
│   └── pyproject.toml
├── backend/                       placeholder for Phase 3 (FastAPI)
├── frontend/                      placeholder for Phase 5 (React)
├── docs/                          architecture notes and labelling guidelines
├── create_structure.py            re-creates the folder skeleton (safe to re-run)
├── requirements.txt
└── .env.example
```

---

## Quick start

### What you need

| Requirement | Notes |
|---|---|
| Git | to clone the repo |
| **Python 3.12** | Use 3.12, not 3.13 or 3.14. Some libraries may not support newer versions yet. |
| [Ollama](https://ollama.com) | runs the model locally |
| **8 GB RAM minimum** | The model needs about 6.4 GB, so close Chrome and other heavy apps before running it. |
| About 6 GB free disk | for the model download |

No GPU is needed. Only `gemma3n:e2b` has been tested.

### 1. Clone and create the environment

Windows (PowerShell):

```powershell
git clone <repo-url>
cd g1resq
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone <repo-url>
cd g1resq
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the `ai_engine` package in editable mode, so `import g1resq_ai` works anywhere and your code changes take effect immediately.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again.

### 2. Install the model

Install Ollama from ollama.com (Windows: `winget install Ollama.Ollama`), open a **new** terminal, then:

```bash
ollama pull gemma3n:e2b
ollama list
```

The download is a few GB. Make sure the Ollama app is running before using the engine.

### 3. Run the tests

```bash
pytest ai_engine/tests -q
```

All tests should pass (59 at the time of writing). They use a fake model backend, so they are instant and do not need Ollama.

### 4. Try the real model

```bash
python ai_engine/scripts/m1_try_reports.py
```

This runs 10 sample reports through the full pipeline (about 3 to 4 minutes; the first report includes the model load). Results are saved to `ai_engine/results/`.

Or use the engine from Python:

```python
from g1resq_ai.pipeline import assess_report

result = assess_report("My father is not breathing at 12 Park Street")
print(result.urgency, result.red_flags)
print(result.summary)
print(result.model_dump_json(indent=2))
```

### Configuration

Defaults live in `ai_engine/g1resq_ai/config.py` and are tuned for an 8 GB CPU-only laptop. A few can be overridden with environment variables (see `.env.example`). A `.env` file is **not** loaded automatically yet, so set variables in your shell.

---

## Useful scripts

| Command | What it does |
|---|---|
| `python ai_engine/scripts/m0_benchmark.py` | Speed, memory and JSON validity of the raw model |
| `python ai_engine/scripts/m1_try_reports.py` | Runs 10 sample reports through the full pipeline |
| `python ai_engine/scripts/m2_speed_diagnosis.py` | Shows where the seconds go (prompt processing vs generation) |
| `python ai_engine/evaluation/gold.py` | Validates the gold test set and shows its class balance |

---

## Contributing

### Workflow

1. Create a branch from `main`: `git checkout -b feature/short-description`
2. Make your change, and add or update tests.
3. Run `pytest ai_engine/tests -q`. Do not open a pull request with failing tests.
4. Open a pull request with a short description of what changed and why. If the change affects AI behaviour, include before and after results.
5. Keep pull requests small and focused, with imperative commit messages ("Add trapped-person rule").

### Rules of the road

- **Prompts are versioned and immutable.** To change a prompt, create `triage_v4.py` with a new `PROMPT_VERSION` and point `pipeline.py` at it. Never edit a prompt that has already produced results.
- **Safety rules can only raise urgency.** Every new rule needs tests: positive cases, and calm sentences that must not trigger it. Over-triage is acceptable; missing a life-threatening report is not.
- **No claims without numbers.** Accuracy, recall and speed claims must come from the evaluation harness on the gold set.
- **The gold set is for measuring, not tuning.** Do not tune prompts against the same reports you report results on. We will introduce a dev/test split once the set is large enough.
- **No real personal data.** Test reports must not contain real names, phone numbers or real home addresses. Invent them.
- **Stay offline-first.** No cloud APIs in the AI engine. The engine must keep working with no internet.
- **Keep the engine web-free.** `ai_engine` must not depend on FastAPI or any web framework, so Phase 3 can wrap it cleanly.
- **Never commit secrets.** `.env` is git-ignored. Commit `.env.example` only.

### Adding reports to the gold test set

The gold set (`ai_engine/data/gold/gold_v0.jsonl`) is how we measure the system. One JSON object per line, UTF-8, no trailing commas:

```json
{"id": "g011", "text": "...", "language": "en", "incident_type": "medical", "urgency": "critical", "people_affected": 1, "location_stated": true, "category": "clear", "borderline": false, "notes": ""}
```

Labelling rules:

| Field | Rule |
|---|---|
| `urgency` | **critical**: life at immediate risk (not breathing, unconscious, heavy bleeding, trapped, people in rising water or fire, collapse with people inside). **high**: serious injury or danger needing a fast response. **medium**: needs a response, no danger to life (e.g. blocked road). **low**: minor, controlled, nobody hurt, or information only. |
| Borderline cases | Label the **higher** level and set `"borderline": true`. |
| Vague reports ("help please") | **high**, type `other`: we cannot rule out a threat to life, so a human must follow up. |
| `incident_type` | What the report describes: `fire`, `flood`, `earthquake`, `accident`, `medical`, `other`. A person falling is `accident`; illness without an accident is `medical`; a tree on a road with no person or vehicle involved is `other`. |
| `people_affected` | People described as hurt or in danger. `0` for "nobody hurt", `null` if not stated. |
| `location_stated` | `true` only for a findable place (address, street name, landmark, area). "Our street", "our flat" and "here" are `false`. |
| `language` | `en`, `hi` (Devanagari), `hinglish` (romanised Hindi), or `mixed`. |
| `category` | `clear`, `vague`, `minor_controlled`, `non_emergency`, `injection`, `noisy`. |

Label what the report **says**, not what you imagine might be true. Run `python ai_engine/evaluation/gold.py` before committing: it reports invalid lines and shows what the set still lacks. Most wanted right now: Hinglish and Hindi reports, and realistic noisy messages (typos, SMS style, missing punctuation).

---

## Known issues (good places to help)

- **Vague reports are over-triaged.** With prompt v3, "help please" can come back as `critical` / `medical` with no supporting evidence.
- **Type hallucination on non-emergencies.** "My phone battery is about to die, send help" is typed `medical`, with an invented `people_affected=1`. The evidence check cannot catch this, because it verifies quotes, not conclusions.
- **`injuries` sometimes holds negative statements** such as "Nobody hurt".
- **Vague places count as locations.** "Our street" and "our flat" are accepted as locations, so `missing_info` can wrongly say location is present.
- **Hindi and Hinglish rule vocabulary is a starter list** in `safety/rules.py` and needs review by native speakers.

---

## Roadmap

1. **AI model development** (current): text assessment, gold set, evaluation harness
2. **AI intelligence:** image and voice reports, local RAG over verified emergency guidelines, duplicate detection, multi-incident summaries
3. **Backend:** FastAPI services, SQLite, authentication, incident management, streaming responses
4. **Offline infrastructure:** local storage, sync when connectivity returns, conflict handling
5. **Frontend:** citizen, responder and command-centre interfaces
6. **Advanced features:** offline maps, resource allocation, predictive analytics
7. **Testing and deployment:** AI accuracy, critical-case recall, offline reliability, security

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` or connection errors | Start the Ollama app and retry. |
| `model not found` | Run `ollama pull gemma3n:e2b`. |
| Very slow responses or a frozen laptop | The model needs about 6.4 GB of RAM. Close browsers and other heavy apps. |
| `pip install` fails on a new library | Check `python --version`. Use 3.12 inside the `.venv`. |
| Activation script blocked in PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| VS Code uses the wrong Python | `Ctrl+Shift+P`, then *Python: Select Interpreter*, then pick the one inside `.venv`. `.vscode/settings.json` points to the Windows path (`.venv/Scripts/python.exe`); on macOS/Linux change it to `.venv/bin/python`. |

---

## Licence and model terms

The project licence has not been chosen yet. Gemma 3n weights are not included in this repository; they are downloaded through Ollama and are subject to Google's Gemma terms of use (open weights, not an OSI open-source licence).

## Acknowledgements

The idea was inspired by [rembertdesigns/gemma3n-disaster-assistant](https://github.com/rembertdesigns/gemma3n-disaster-assistant). G1RESQ is written from scratch and uses that repository only as a reference for ideas.