"""Measure streaming completion latency through an oMLX endpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.request import Request, urlopen


def measure(url, payload, timeout, *, skip_cache_store=False):
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if skip_cache_store:
        headers["X-oMLX-Cache-Store"] = "off"
    if token := os.getenv("OMLX_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, json.dumps(payload).encode(), headers)
    start = time.perf_counter()
    first_output = None
    usage = {}
    output = hashlib.sha256()
    completed = False
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                completed = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError("Server reported a streaming error")
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                fragment = choice.get("text") or delta.get("content") or delta.get("reasoning_content")
                if fragment:
                    if first_output is None:
                        first_output = time.perf_counter() - start
                    output.update(fragment.encode("utf-8"))
            if event.get("usage"):
                usage = {key: event["usage"].get(key) for key in
                         ("prompt_tokens", "completion_tokens")}
                cached = event["usage"].get("cached_tokens")
                if cached is None:
                    cached = (event["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                usage["cached_tokens"] = cached
    if not completed or first_output is None:
        raise ValueError("Stream ended without output and a completion marker")
    return {"ttft_seconds": first_output, "elapsed_seconds": time.perf_counter() - start,
            "output_sha256": output.hexdigest(), **usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--condition", required=True, help="Record helper, layers, chunk and cache settings")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--skip-cache-store", action="store_true",
                        help="Ask oMLX not to cache this request")
    args = parser.parse_args()
    if min(args.runs, args.max_tokens, args.timeout) <= 0 or args.warmups < 0:
        parser.error("runs, max-tokens and timeout must be positive, warmups non-negative")
    data = args.prompt.read_bytes()
    payload = {"model": args.model, "prompt": data.decode("utf-8"), "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0,
               "max_tokens": args.max_tokens}
    common = {"model": args.model, "revision": args.revision, "condition": args.condition,
              "prompt_sha256": hashlib.sha256(data).hexdigest(), "max_tokens": args.max_tokens,
              "skip_cache_store": args.skip_cache_store}
    samples = []
    failures = 0
    with args.output.open("x", encoding="utf-8") as log:
        for index in range(args.warmups + args.runs):
            warmup = index < args.warmups
            row = {**common, "iteration": index + 1, "warmup": warmup}
            try:
                row.update(measure(args.url, payload, args.timeout,
                                   skip_cache_store=args.skip_cache_store))
                row["status"] = "ok"
                if not warmup:
                    samples.append(row)
            except (OSError, ValueError) as error:
                row.update(status="error", error_type=type(error).__name__)
                failures += 1
            print(json.dumps(row), file=log, flush=True)
        summary = {"measured_ok": len(samples), "failed_including_warmups": failures}
        for key in ("ttft_seconds", "elapsed_seconds"):
            if samples:
                summary[f"median_{key}"] = statistics.median(row[key] for row in samples)
        print(json.dumps(summary))
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
