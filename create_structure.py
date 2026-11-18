"""Create the G1RESQ project skeleton.

Safe to re-run: it never overwrites an existing file, so your
m0_benchmark.py and results are untouched.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q = '"' * 3


def stub(description, milestone):
    return f"{Q}{description}\n\nStatus: placeholder. Implemented in {milestone}.\n{Q}\n"


GITIGNORE = """\
# Python
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/

# Secrets and local config
.env

# Local data and models
*.db
*.sqlite3
ai_engine/data/raw/
ai_engine/results/*.json
!ai_engine/results/.gitkeep

# OS / editor
.DS_Store
Thumbs.db
"""

ENV_EXAMPLE = """\
OLLAMA_URL=http://localhost:11434
MODEL_NAME=gemma3n:e2b
NUM_CTX=2048
NUM_PREDICT=200
REQUEST_TIMEOUT_SEC=300
"""

PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "g1resq-ai"
version = "0.1.0"
description = "Offline-first emergency report assessment engine"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.7", "requests>=2.31"]

[tool.setuptools.packages.find]
include = ["g1resq_ai*"]
"""

VSCODE_SETTINGS = """\
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["ai_engine/tests"]
}
"""

README = """\
# G1RESQ

Offline-first multimodal AI disaster response and emergency coordination system.

- `ai_engine/`  Phase 1: Gemma 3n assessment engine (importable package)
- `backend/`    Phase 3: FastAPI services
- `frontend/`   Phase 5: citizen, responder and command interfaces
- `docs/`       architecture notes and labeling guidelines

AI output is decision support only. A human makes every triage and dispatch decision.
"""

DIRS = [
    "ai_engine/data/gold",
    "ai_engine/data/raw",
    "ai_engine/results",
    "ai_engine/scripts",
    "frontend",
    "docs",
    ".vscode",
]

PKG = "ai_engine/g1resq_ai"

FILES = {
    # ---- root ----
    "README.md": README,
    ".gitignore": GITIGNORE,
    ".env.example": ENV_EXAMPLE,
    "requirements.txt": "-e ./ai_engine\npytest>=8.0\n",
    ".vscode/settings.json": VSCODE_SETTINGS,
    # ---- docs ----
    "docs/architecture.md": "# Architecture\n\nSee the Phase 1 design. Update as decisions change.\n",
    "docs/labeling_guidelines.md": "# Labeling guidelines\n\nHow to label the gold test set (written in M3).\n",
    # ---- ai_engine package ----
    "ai_engine/pyproject.toml": PYPROJECT,
    f"{PKG}/__init__.py": stub("G1RESQ AI engine: offline emergency report assessment.", "M4"),
    f"{PKG}/config.py": stub("Settings: model name, Ollama URL, context size, timeouts.", "M1"),
    f"{PKG}/schemas.py": stub("Pydantic models, including IncidentAssessment.", "M1"),
    f"{PKG}/parsing.py": stub("Parse and validate model output; retry and safe fallback.", "M1"),
    f"{PKG}/pipeline.py": stub("assess_report(text, metadata) -> IncidentAssessment.", "M1 to M4"),
    f"{PKG}/backends/__init__.py": "",
    f"{PKG}/backends/base.py": stub("Abstract LLM backend interface.", "M1"),
    f"{PKG}/backends/ollama_backend.py": stub("Ollama HTTP backend.", "M1"),
    f"{PKG}/safety/__init__.py": "",
    f"{PKG}/safety/rules.py": stub("Rule-based urgency floor (red-flag keywords).", "M2"),
    f"{PKG}/prompts/__init__.py": "",
    f"{PKG}/prompts/triage_v1.py": stub("Versioned triage prompt.", "M1"),
    # ---- evaluation and tests ----
    "ai_engine/evaluation/__init__.py": "",
    "ai_engine/evaluation/run_eval.py": stub("Run the test set and report metrics.", "M3"),
    "ai_engine/tests/__init__.py": "",
    "ai_engine/tests/test_schemas.py": stub("Schema validation tests.", "M1"),
    "ai_engine/tests/test_rules.py": stub("Safety rule tests.", "M2"),
    # ---- backend placeholder (Phase 3) ----
    "backend/app/__init__.py": "",
    "backend/app/main.py": stub("FastAPI entry point.", "Phase 3"),
    "backend/app/api/__init__.py": "",
    "backend/app/models/__init__.py": "",
    "backend/app/services/__init__.py": "",
    "backend/tests/__init__.py": "",
    # ---- keep empty folders in git ----
    "ai_engine/data/gold/.gitkeep": "",
    "ai_engine/results/.gitkeep": "",
    "frontend/.gitkeep": "",
}


def main():
    created, skipped = 0, 0
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"exists   {rel}")
            skipped += 1
        else:
            path.write_text(content, encoding="utf-8")
            print(f"created  {rel}")
            created += 1
    print(f"\nDone. {created} created, {skipped} already existed.")


if __name__ == "__main__":
    main()