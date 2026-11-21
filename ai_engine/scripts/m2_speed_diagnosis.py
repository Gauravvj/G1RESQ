"""M2 speed diagnosis: where do the seconds go?

Sends the real triage prompt to Ollama under several settings and prints
Ollama's own timing counters. Changes no project code.

Each setting runs twice with DIFFERENT reports but the same system prompt.
If the second run evaluates far fewer prompt tokens than the first, Ollama is
reusing the cached system prompt.
"""
import requests

from g1resq_ai.config import get_settings
from g1resq_ai.prompts.triage_v2 import build_messages
from g1resq_ai.schemas import llm_output_schema

REPORTS = [
    "Water is rising fast in our street, about 6 people on the roof of a house, one is an elderly woman who cannot walk.",
    "Car hit a motorbike near the flyover, two people lying on the road, one is bleeding from the head.",
]
SCHEMA = llm_output_schema()

VARIANTS = [
    ("A  schema, default threads", SCHEMA, {}),
    ("B  plain json (no schema), default threads", "json", {}),
    ("C  schema, 4 threads", SCHEMA, {"num_thread": 4}),
    ("D  schema, 8 threads", SCHEMA, {"num_thread": 8}),
]


def call(fmt, extra_options, report):
    s = get_settings()
    payload = {
        "model": s.model_name,
        "messages": build_messages(report),
        "stream": False,
        "format": fmt,
        "keep_alive": s.keep_alive,
        "options": {"temperature": 0, "num_ctx": s.num_ctx, "num_predict": s.num_predict, **extra_options},
    }
    r = requests.post(f"{s.ollama_url}/api/chat", json=payload, timeout=s.request_timeout_sec)
    r.raise_for_status()
    return r.json()


def secs(d, key):
    return d.get(key, 0) / 1e9


def main():
    print("Warm-up (loads the model)...")
    call(SCHEMA, {}, REPORTS[0])
    print(f"{'setting / run':<54}{'load':>6}{'prompt tok':>11}{'prompt s':>9}{'out tok':>8}{'gen s':>7}{'tok/s':>7}{'total':>7}")
    for label, fmt, opts in VARIANTS:
        for i, report in enumerate(REPORTS, 1):
            d = call(fmt, opts, report)
            gen = secs(d, "eval_duration")
            tps = d.get("eval_count", 0) / gen if gen else 0.0
            print(
                f"{label + ' / run ' + str(i):<54}"
                f"{secs(d, 'load_duration'):>6.1f}"
                f"{d.get('prompt_eval_count', 0):>11}"
                f"{secs(d, 'prompt_eval_duration'):>9.1f}"
                f"{d.get('eval_count', 0):>8}"
                f"{gen:>7.1f}"
                f"{tps:>7.1f}"
                f"{secs(d, 'total_duration'):>7.1f}"
            )


if __name__ == "__main__":
    main()