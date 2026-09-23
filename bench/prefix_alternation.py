#!/usr/bin/env python3
"""Prefix reuse under alternating conversations — reproducer and A/B arm.

Two independent long conversations (A and B), advanced in strict alternation.
For A, every one of B's turns is interleaved traffic, and vice versa. Each turn
is its own request, which is what an agentic loop does.

The pass criterion is not "a high hit rate" but "this turn reused the whole
prefix the previous turn established": ``cached_tokens >= 0.9 * prev_prompt``.
Failure is reported as PREFIX-LOST and means the turn re-prefilled in full.

Regimes are sharp, so read the size table in the issue rather than one run:

    one conversation, no second one     ~99.7% reused
    two conversations, sequential       ~96.6-99.7%
    two conversations, alternating      0% every turn  (above a size knee)

Usage:
    python bench/prefix_alternation.py --target-tokens 60000 --rounds 3
    python bench/prefix_alternation.py --target-tokens 96000 --noise 2 --rounds 4
    HQ_UNIT=qwen38-hq-vllm python bench/prefix_alternation.py    # engine-env label
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(path):  # same convention as quality_battery.py
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
UNIT = os.environ.get("HQ_UNIT", "qwen38-hq-vllm")

LOG = []


def call(messages, max_tokens=24):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        API + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    el = time.time() - t0
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    return {"prompt": u["prompt_tokens"], "cached": det.get("cached_tokens", 0),
            "gen": u["completion_tokens"], "secs": el,
            "text": (d["choices"][0]["message"].get("content") or "").strip()}


FILLERS = {
    "A": ("The scheduler interleaves prefill chunks with decode steps. "
          "A Mamba state snapshot is materialised at the last prefill chunk boundary. "
          "Attention prefix caching is the intersection of per-group hit sets. "),
    "B": ("Quantisation geometry decides how many tokens fit in the pool. "
          "Speculative decoding acceptance is what turns steps into throughput. "
          "Chunked prefill bounds how long a co-tenant can be starved. "),
    "N": ("Unrelated interleaved traffic allocates blocks between turns. "
          "This text exists only to consume KV blocks and force eviction order. "),
}


def make_doc(side, n_chars):
    filler = FILLERS[side]
    body = filler * (n_chars // len(filler) + 1)
    lines = [body[i:i + 900] for i in range(0, n_chars, 900)]
    return "\n".join(f"[{side}{i:05d}] {ln}" for i, ln in enumerate(lines))[:n_chars]


def log(rec):
    LOG.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def engine_env():
    """Read the ENGINE process's environment, not this shell's.

    Otherwise an arm label is whatever the caller happened to export, which is
    how a run gets reported against the wrong arm. Returns the retention setting
    and max_model_len, or placeholders if the unit cannot be read.
    """
    try:
        pid = subprocess.check_output(
            ["systemctl", "show", UNIT, "-p", "MainPID", "--value"],
            text=True, timeout=10).strip()
        raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
        d = dict(x.split("=", 1) for x in raw.split("\0") if "=" in x)
        extra = d.get("EXTRA_ARGS", "")
        ret = "<unset>"
        for tok, nxt in zip(extra.split(), extra.split()[1:]):
            if tok == "--prefix-cache-retention-interval":
                ret = nxt
        if ret == "<unset>":
            ret = d.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "<unset>")
        return ret, d.get("MAX_LEN", "?")
    except Exception as exc:  # noqa: BLE001
        return f"<err:{exc}>", "?"


def main():
    ap = argparse.ArgumentParser(
        description="Prefix reuse under alternating conversations (A/B arm).")
    ap.add_argument("--target-tokens", type=int, default=100000,
                    help="base document size per conversation, in tokens")
    ap.add_argument("--rounds", type=int, default=8, help="turns per conversation")
    ap.add_argument("--noise", type=int, default=2,
                    help="unrelated requests injected between turns")
    ap.add_argument("--noise-tokens", type=int, default=4000)
    ap.add_argument("--budget-min", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    retention, maxlen = engine_env()
    tag = f"RET{retention}_ML{maxlen}"
    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    out = args.out or os.path.join(
        HERE, f"prefix-alternation-{tag}-{dt.datetime.now():%Y%m%d-%H%M%S}.json")
    print(f"══ alternating-conversation prefix reuse ══  engine: retention={retention} "
          f"MAX_LEN={maxlen}  (read from /proc/<engine>, not this shell)")
    print(f"   {args.target_tokens:,} tok/conversation  rounds={args.rounds}  "
          f"noise={args.noise}x{args.noise_tokens} tok  budget={args.budget_min:.0f} min")
    print(f"   output -> {out}\n")

    # Calibrate tokens/char against THIS stack's tokenizer; a hardcoded ratio
    # silently changes the arm you think you ran by tens of percent.
    cal = call([{"role": "user", "content": "CALIB\n" + make_doc("A", 20000)}], max_tokens=1)
    ratio = cal["prompt"] / 20000.0
    n_chars = int(args.target_tokens / ratio)
    print(f"calibration: 20,000 chars -> {cal['prompt']:,} tok ({ratio:.4f} tok/char)")
    print(f"per-conversation document: {n_chars:,} chars ≈ {args.target_tokens:,} tok\n")
    log({"event": "calib", "ratio": ratio, "n_chars": n_chars, "arm": tag})

    docs = {s: make_doc(s, n_chars) for s in ("A", "B")}
    conv = {s: [{"role": "user", "content": docs[s]}] for s in ("A", "B")}
    prev_prompt = {s: 0 for s in ("A", "B")}
    stats = {"reqs": 0, "lost": 0, "healthy": 0, "cold": 0}
    fail_rows = []

    for rnd in range(1, args.rounds + 1):
        for side in ("A", "B"):
            if time.time() > deadline:
                print(f"\nbudget exhausted before round {rnd} {side}")
                rnd = args.rounds + 1
                break
            q = (f"This is {side}'s question for round {rnd}. Answer in one sentence, "
                 f"do not restate the document.")
            conv[side].append({"role": "user", "content": q})
            res = call(conv[side], max_tokens=24)
            conv[side].append({"role": "assistant", "content": res["text"] or "OK"})

            stats["reqs"] += 1
            if rnd == 1:
                stats["cold"] += 1
                verdict = "COLD"
            else:
                need = 0.9 * prev_prompt[side]
                if res["cached"] < need:
                    stats["lost"] += 1
                    verdict = "PREFIX-LOST"
                    fail_rows.append((rnd, side, prev_prompt[side], res["cached"], res["secs"]))
                else:
                    stats["healthy"] += 1
                    verdict = "ok"
            pct = (res["cached"] / res["prompt"] * 100) if res["prompt"] else 0
            print(f"  r{rnd:>2} {side}  {verdict:<11} "
                  f"prompt={res['prompt']:>7,}  cached={res['cached']:>7,} ({pct:5.1f}%)  "
                  f"{res['secs']:6.2f}s")
            log({"event": "turn", "round": rnd, "side": side, "verdict": verdict,
                 "prompt": res["prompt"], "cached": res["cached"], "pct": round(pct, 1),
                 "secs": round(res["secs"], 2), "prev_prompt": prev_prompt[side]})
            prev_prompt[side] = res["prompt"]

            for k in range(args.noise):
                if time.time() > deadline:
                    break
                nd = f"NOISE-{side}-{rnd}-{k}-{time.time():.6f}\n" + make_doc(
                    "N", int(args.noise_tokens / ratio))
                nr = call([{"role": "user", "content": nd}], max_tokens=4)
                log({"event": "noise", "round": rnd, "side": side, "k": k,
                     "prompt": nr["prompt"], "cached": nr["cached"],
                     "secs": round(nr["secs"], 2)})
        if time.time() > deadline:
            break

    el = time.time() - t_start
    print("\n" + "=" * 84)
    print(f"arm={tag}  {el:.0f}s  conversation requests {stats['reqs']} "
          f"(cold {stats['cold']} / healthy {stats['healthy']} / **lost {stats['lost']}**)")
    if fail_rows:
        print("\nprefix reuse lost:")
        for r, s, pp, c, sec in fail_rows:
            print(f"  r{r} {s}  previous prefix {pp:,} tok, reused {c:,} tok "
                  f"(short by {pp - c:,}, {sec:.1f}s)")
        print("\nthis arm REPRODUCES the defect.")
    else:
        print("\n✅ no prefix loss: every turn reused the previous turn's whole prefix.")
        print("   ⚠️ This only says so for this arm. To claim a fix, run the dense")
        print("      arm too — if that is also clean, the test failed to reproduce")
        print("      rather than the fix working.")
    with open(out, "w") as f:
        json.dump({"arm": tag, "args": vars(args), "stats": stats,
                   "elapsed_s": el, "log": LOG}, f, ensure_ascii=False, indent=1)
    print(f"\nraw records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
