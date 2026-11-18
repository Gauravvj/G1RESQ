"""M0: measure Gemma 3n E2B speed, memory and JSON validity through Ollama.

Uses only the Python standard library, so it runs on any Python 3.
"""
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"
MODEL = "gemma3n:e2b"

SYSTEM = (
    "You are an emergency report triage assistant. "
    "Reply with ONLY a JSON object with these keys: "
    '"type" (one of: fire, flood, earthquake, accident, medical, other), '
    '"urgency" (one of: critical, high, medium, low), '
    '"people_affected" (integer or null), '
    '"location" (string or null), '
    '"summary" (maximum 15 words).'
)

REPORTS = [
    "My father is not breathing and has gone blue. Please send an ambulance "
    "to 12 Park Street, third floor.",
    "Small kitchen fire in our flat, we put it out with water. Nobody hurt.",
    "Bhai papa ko saans nahi aa rahi, jaldi ambulance bhejo, gali number 4 "
    "mandir ke paas.",
    "Water is rising fast in our street, about 6 people on the roof of a house, "
    "one is an elderly woman who cannot walk.",
]


def call(path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        OLLAMA + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(report):
    return call(
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 150},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": report},
            ],
        },
    )


def secs(ns):
    return ns / 1e9


def main():
    try:
        tags = call("/api/tags")
    except (urllib.error.URLError, ConnectionError):
        sys.exit("Ollama is not running. Start it (open the Ollama app) and retry.")

    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.startswith(MODEL) for n in names):
        sys.exit(f"Model {MODEL} not found. Run: ollama pull {MODEL}")

    print(f"Model: {MODEL}")
    print("Warm-up (loads the model into RAM, may take a minute)...")
    warm = chat("test")
    print(f"  load time: {secs(warm.get('load_duration', 0)):.1f}s\n")

    rows = []
    for i, report in enumerate(REPORTS, 1):
        r = chat(report)
        text = r["message"]["content"]
        try:
            parsed = json.loads(text)
            valid = True
        except json.JSONDecodeError:
            parsed, valid = None, False

        gen_s = secs(r["eval_duration"])
        tps = r["eval_count"] / gen_s if gen_s else 0.0
        pre_s = secs(r["prompt_eval_duration"])
        pps = r["prompt_eval_count"] / pre_s if pre_s else 0.0
        total = secs(r["total_duration"])
        rows.append({"report": report, "valid": valid, "tokens_per_sec": tps,
                     "prompt_tokens_per_sec": pps, "total_sec": total,
                     "output": parsed if valid else text})

        print(f"--- Report {i} ---")
        print(report)
        print(f"JSON valid: {valid}")
        print(json.dumps(parsed, ensure_ascii=False) if valid else text)
        print(f"generation: {tps:.1f} tok/s | prompt: {pps:.1f} tok/s | "
              f"total: {total:.1f}s\n")

    ps = call("/api/ps")
    mem_gb = max((m.get("size", 0) for m in ps.get("models", [])), default=0) / 1e9

    print("=== SUMMARY ===")
    print(f"JSON valid:           {sum(r['valid'] for r in rows)}/{len(rows)}")
    print(f"Median generation:    {statistics.median(r['tokens_per_sec'] for r in rows):.1f} tok/s")
    print(f"Median total time:    {statistics.median(r['total_sec'] for r in rows):.1f}s per report")
    print(f"Model memory in use:  {mem_gb:.1f} GB")

    out = Path(__file__).resolve().parent.parent / "results" / "m0_results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()