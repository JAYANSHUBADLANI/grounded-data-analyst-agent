import json
import time
import statistics
import os
from dotenv import load_dotenv
from baseline import run_baseline
from agent import run_agent
from verification import verify_grounding, extract_numbers
from llm_client import PROVIDER, MODEL_NAME, PRICE_PER_MILLION_INPUT_TOKENS, PRICE_PER_MILLION_OUTPUT_TOKENS

load_dotenv()

def check_success(final_answer: str, ground_truth: str) -> bool:
    if ground_truth == "None" or ground_truth == "ERROR":
        return False
    
    ans_nums = extract_numbers(final_answer)
    gt_nums = extract_numbers(ground_truth)
    
    if not gt_nums:
        return ground_truth.lower() in final_answer.lower()
        
    gt_num = gt_nums[0]
    if gt_num in ans_nums:
        return True
    
    try:
        gt_float = float(gt_num)
        for num in ans_nums:
            try:
                if float(num) == gt_float:
                    return True
            except ValueError:
                pass
    except ValueError:
        pass
        
    return False

def _fresh_stats():
    return {
        "attempted": 0,
        "successes": 0,
        "latencies": [],
        "tool_calls": [],
        "retries": [],
        "grounded_checked": 0,
        "hallucination_triggers": 0,
        "input_tokens": 0,
        "output_tokens": 0
    }


def run_evaluation(json_path="benchmark_qa.json"):
    print(f"Loading benchmark from {json_path}...")
    with open(json_path, 'r') as f:
        qa_data = json.load(f)

    results = {"baseline": _fresh_stats(), "agent": _fresh_stats()}

    total_q = len(qa_data)
    evaluation_log = []
    stopped_early = False
    stop_reason = None

    for i, qa in enumerate(qa_data):
        print(f"\n[{i+1}/{total_q}] Q: {qa['question']}")
        print(f"Ground Truth: {qa['ground_truth_answer']}")

        print("  Running Baseline...")
        b_res = run_baseline(qa["question"])
        b_error = b_res.get("error_type")

        if b_error == "rate_limited":
            print(f"    Rate limited. Stopping evaluation after {i} of {total_q} questions.")
            stopped_early = True
            stop_reason = "rate_limited"
            break

        # An error string is not a real answer. Score it as a plain failure and
        # skip the number-matching / grounding checks instead of running them
        # against error text, which can spuriously match on stray digits.
        if b_error:
            b_success = False
            b_grounded = None
        else:
            b_success = check_success(b_res["final_answer"], qa["ground_truth_answer"])
            b_grounded, _ = verify_grounding(b_res["final_answer"], b_res["steps"])

        b_in_tok = b_res.get("input_tokens", 0)
        b_out_tok = b_res.get("output_tokens", 0)

        results["baseline"]["attempted"] += 1
        results["baseline"]["successes"] += 1 if b_success else 0
        results["baseline"]["latencies"].append(b_res["latency_seconds"])
        results["baseline"]["tool_calls"].append(b_res["tool_calls_count"])
        results["baseline"]["input_tokens"] += b_in_tok
        results["baseline"]["output_tokens"] += b_out_tok
        if b_grounded is not None:
            results["baseline"]["grounded_checked"] += 1
            results["baseline"]["hallucination_triggers"] += 1 if not b_grounded else 0

        print(f"    Success: {b_success} | Latency: {b_res['latency_seconds']}s")
        print(f"    Tokens: {b_in_tok} in, {b_out_tok} out")

        print("  Running Agent...")
        a_res = run_agent(qa["question"])
        a_error = a_res.get("error_type")

        if a_error == "rate_limited":
            print(f"    Rate limited. Stopping evaluation after {i + 1} baseline / {i} agent questions.")
            evaluation_log.append({
                "question": qa["question"],
                "ground_truth": qa["ground_truth_answer"],
                "baseline": {
                    "answer": b_res["final_answer"],
                    "steps": b_res["steps"],
                    "success": b_success,
                    "latency": b_res["latency_seconds"],
                    "input_tokens": b_in_tok,
                    "output_tokens": b_out_tok
                },
                "agent": {"skipped": "rate_limited"}
            })
            stopped_early = True
            stop_reason = "rate_limited"
            break

        if a_error:
            a_success = False
            a_grounded, a_grounding_reason = None, []
        else:
            a_success = check_success(a_res["final_answer"], qa["ground_truth_answer"])
            a_grounded, a_grounding_reason = verify_grounding(a_res["final_answer"], a_res["steps"])

        a_in_tok = a_res.get("input_tokens", 0)
        a_out_tok = a_res.get("output_tokens", 0)

        results["agent"]["attempted"] += 1
        results["agent"]["successes"] += 1 if a_success else 0
        results["agent"]["latencies"].append(a_res["latency_seconds"])
        results["agent"]["tool_calls"].append(a_res["tool_calls_count"])
        results["agent"]["retries"].append(a_res["retry_count"])
        results["agent"]["input_tokens"] += a_in_tok
        results["agent"]["output_tokens"] += a_out_tok
        if a_grounded is not None:
            results["agent"]["grounded_checked"] += 1
            results["agent"]["hallucination_triggers"] += 1 if not a_grounded else 0

        print(f"    Success: {a_success} | Latency: {a_res['latency_seconds']}s | Retries: {a_res['retry_count']}")
        print(f"    Tokens: {a_in_tok} in, {a_out_tok} out")

        evaluation_log.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth_answer"],
            "baseline": {
                "answer": b_res["final_answer"],
                "steps": b_res["steps"],
                "success": b_success,
                "latency": b_res["latency_seconds"],
                "input_tokens": b_in_tok,
                "output_tokens": b_out_tok,
                "error_type": b_error
            },
            "agent": {
                "answer": a_res["final_answer"],
                "steps": a_res["steps"],
                "success": a_success,
                "latency": a_res["latency_seconds"],
                "input_tokens": a_in_tok,
                "output_tokens": a_out_tok,
                "retries": a_res["retry_count"],
                "error_type": a_error,
                "grounding_result": {
                    "is_grounded": a_grounded,
                    "reason": a_grounding_reason
                } if a_grounded is not None else None
            }
        })

    with open("evaluation_log.json", "w") as f:
        json.dump(evaluation_log, f, indent=2)

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)

    if stopped_early:
        print(
            f"\nSTOPPED EARLY ({stop_reason}) after {results['baseline']['attempted']} of "
            f"{total_q} questions. Every statistic below is computed only over the "
            f"questions actually attempted, none of it is diluted by unattempted ones."
        )

    for mode in ["baseline", "agent"]:
        n = results[mode]["attempted"]
        print(f"\n*** {mode.upper()} *** ({n} of {total_q} questions attempted)")
        if n == 0:
            print("No questions attempted.")
            continue

        lat = results[mode]["latencies"]
        print(f"Success Rate: {(results[mode]['successes'] / n) * 100:.1f}%")
        print(f"Avg Latency: {statistics.mean(lat):.2f}s  (median {statistics.median(lat):.2f}s)")
        print(f"Avg Tool Calls: {statistics.mean(results[mode]['tool_calls']):.2f}")

        gc = results[mode]["grounded_checked"]
        if gc:
            print(f"Hallucination Guard Trigger Rate: {(results[mode]['hallucination_triggers'] / gc) * 100:.1f}% (of {gc} answers actually graded)")
        else:
            print("Hallucination Guard Trigger Rate: n/a, no answers were gradeable")

        in_tok = results[mode]["input_tokens"]
        out_tok = results[mode]["output_tokens"]
        print(f"Total Tokens: {in_tok} in / {out_tok} out")
        print(f"Avg Tokens/Question: {in_tok / n:.1f} in / {out_tok / n:.1f} out")
        cost = (in_tok * PRICE_PER_MILLION_INPUT_TOKENS + out_tok * PRICE_PER_MILLION_OUTPUT_TOKENS) / 1_000_000
        provider_label = "Groq Free Tier" if PROVIDER == "groq" else f"{MODEL_NAME} on Vertex AI"
        print(f"Estimated Cost: ${cost:.4f} ({provider_label})")

        if mode == "agent":
            print(f"Avg Retries: {statistics.mean(results[mode]['retries']):.2f}")
            retry_rate = (sum(1 for r in results[mode]['retries'] if r > 0) / n) * 100
            print(f"Questions with at least one retry: {retry_rate:.1f}%")

    print("="*40)

if __name__ == "__main__":
    run_evaluation()
