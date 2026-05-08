"""One-shot local CLI runner for notebook transport comparisons.

This module is executed as a subprocess by `via_cli_local.run(...)`:

    python -m mcp_demo.clients.local_cli_runner --question "..." --model "..."

It intentionally avoids the heavy Week-1 retriever stack so the comparison can
still run in constrained environments. The point of this client is to benchmark
the *CLI transport shape* (process spawn + stdout parsing), not retrieval quality.
"""
from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage, SystemMessage

from shared import estimate_cost, get_llm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--model", default="openai/gpt-5.4-nano")
    args = p.parse_args()

    llm = get_llm(args.model)
    out = llm.invoke([
        SystemMessage(content=(
            "You are a non-interactive CLI assistant. Answer the user's RAG-related "
            "question concisely and clearly. If uncertain, say so."
        )),
        HumanMessage(content=args.question),
    ])
    usage = getattr(out, "usage_metadata", None) or {}
    in_t = int(usage.get("input_tokens", 0) or 0)
    out_t = int(usage.get("output_tokens", 0) or 0)
    payload = {
        "answer": out.content if isinstance(out.content, str) else str(out.content),
        "n_tool_calls": 0,
        "tool_latency_total_s": 0.0,
        "total_latency_s": 0.0,   # wrapper measures process+model time end-to-end
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_usd": estimate_cost(args.model, in_t, out_t),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

