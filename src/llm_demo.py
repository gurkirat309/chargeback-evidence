"""
Phase 4 — LLM pipeline demo + metrics (CLAUDE.md section 11).

NOTE: not in the section 14 layout — the demo/metrics driver for the LLM layer
(the harness proper is Phase 5). Runs evidence agent -> generator -> verifier ->
packet on a sample of disputes and reports:
  * citation coverage %  (generated claims whose artifact_id resolves)
  * hallucination rate under stress (delete a cited artifact, regenerate, measure
    how often the deleted fact is asserted anyway — before vs after verification)

All LLM responses are cached, so re-runs need no network.
"""
from __future__ import annotations

import argparse
import copy
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # model output may contain non-cp1252 chars
except Exception:
    pass

import numpy as np
import pandas as pd

import evidence_agent as EA
import llm_generator as GEN
import llm_verifier as VER
from features import PROC

# distinctive keywords per artifact label, for the stress-test fact detector
LABEL_KEYWORDS = {
    "AVS result": ["avs", "address verification"],
    "CVV result": ["cvv", "security code"],
    "3-D Secure": ["3-d secure", "3ds", "3-d", "three-d secure"],
    "Account tenure": ["account age", "account tenure"],
    "Delivery proof": ["deliver"],
    "Device match": ["device"],
    "IP origin": ["datacenter", "residential ip", "ip address"],
    "Prior dispute record": ["undisputed"],
    "Support log": ["support", "ticket"],
    "Consent record": ["consent"],
    "Product photos": ["photo"],
    "Refund policy": ["refund policy"],
    "Agent mandate": ["mandate"],
}


def load_rows():
    disputes = pd.read_parquet(PROC / "disputes.parquet").set_index("dispute_id")
    evidence = pd.read_parquet(PROC / "evidence.parquet").set_index("dispute_id")
    preds = pd.read_parquet(PROC / "predictions.parquet").set_index("dispute_id")
    return disputes, evidence, preds


def run_pipeline(did, disputes, evidence):
    bundle = EA.build_bundle(disputes.loc[did].to_dict() | {"dispute_id": did},
                             evidence.loc[did])
    gen = GEN.generate_letter(bundle)
    verified = VER.verify_claims(gen["claims"], bundle)
    packet = VER.build_packet(bundle, gen["framing"], verified)
    return bundle, gen, verified, packet


def citation_coverage(verified_all):
    n = sum(len(v) for v in verified_all)
    resolves = sum(1 for v in verified_all for c in v if c["resolves"])
    return resolves / n if n else float("nan"), resolves, n


def _asserts_fact(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


def stress_one(did, disputes, evidence, rng):
    """Delete one cited FAVOURABLE artifact, regenerate, check if the deleted fact
    reappears — before and after verification."""
    bundle, gen, verified, packet = run_pipeline(did, disputes, evidence)
    # candidate = a cited artifact that is favourable and not the step-1 order record
    cited = [c["artifact_id"] for c in gen["claims"]]
    cands = [aid for aid in cited
             if bundle.resolve(aid) and bundle.resolve(aid).favourable
             and bundle.resolve(aid).step != 1
             and bundle.resolve(aid).label in LABEL_KEYWORDS]
    if not cands:
        return None
    victim = cands[rng.integers(len(cands))]
    label = bundle.resolve(victim).label
    keywords = LABEL_KEYWORDS[label]

    reduced = copy.deepcopy(bundle)
    reduced.artifacts.pop(victim)
    for s in reduced.steps:
        s["artifact_ids"] = [a for a in s["artifact_ids"] if a != victim]

    gen2 = GEN.generate_letter(reduced)
    verified2 = VER.verify_claims(gen2["claims"], reduced)
    packet2 = VER.build_packet(reduced, gen2["framing"], verified2)

    raw_text = " ".join(c["claim"] for c in gen2["claims"])
    kept_text = " ".join(c["claim"] for c in packet2["claims"])
    return {
        "deleted_label": label,
        "asserted_pre": _asserts_fact(raw_text, keywords),
        "asserted_post": _asserts_fact(kept_text, keywords),
    }


def adversarial_check(disputes, evidence, preds):
    """Prove the verifier is not a rubber stamp: fabricate claims that CONTRADICT
    their cited artifact and confirm the verifier strips them. Searches disputes
    for real unfavourable artifacts (AVS no-match, device mismatch, datacenter IP)."""
    contradiction = {
        "AVS result": "The billing address was fully verified and matched (AVS match).",
        "Device match": "The device fingerprint matched the customer's history exactly.",
        "IP origin": "The order originated from the customer's home residential IP.",
        "3-D Secure": "The transaction was fully 3-D Secure authenticated and passed.",
    }
    tried, caught = 0, 0
    for did in preds.index:
        b = EA.build_bundle(disputes.loc[did].to_dict() | {"dispute_id": did},
                            evidence.loc[did])
        for a in b.artifacts.values():
            if a.label in contradiction and not a.favourable and tried < 5:
                false_claim = {"claim": contradiction[a.label], "artifact_id": a.artifact_id}
                v = VER.verify_claims([false_claim], b)[0]
                tried += 1
                stripped = not v["supported"]
                caught += stripped
                print(f"  [{'CAUGHT' if stripped else 'MISSED'}] artifact says: "
                      f"\"{a.statement}\"")
                print(f"            false claim : \"{false_claim['claim']}\"")
                print(f"            verifier    : supported={v['supported']}  ({v['reason']})")
        if tried >= 5:
            break
    print(f"\n  adversarial: verifier caught {caught}/{tried} fabricated claims")


def print_example(did, bundle, gen, verified, packet):
    print("=" * 72)
    print(f"EXAMPLE LETTER PACKET — {did}  (reason: {bundle.reason_code})")
    print("=" * 72)
    print("\nEvidence agent — 8-step trace:")
    for s in bundle.steps:
        n = len(s["artifact_ids"])
        print(f"  [{s['ts_ms']:>4}ms] step {s['step']} {s['name']:<28} "
              f"{n} artifact(s)")
    print(f"\nFraming: {packet['framing']}")
    print(f"\nGenerated {gen['claims'].__len__()} claims; verifier kept "
          f"{packet['n_kept']}:")
    for v in verified:
        mark = "KEEP" if (v["resolves"] and v["supported"]) else "STRIP"
        print(f"  [{mark}] ({v['artifact_id']}) {v['claim']}")
        if mark == "STRIP":
            print(f"         reason: {v['reason']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="disputes to sample")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    disputes, evidence, preds = load_rows()
    pool = preds[preds["split"] == args.split].index.tolist()
    rng = np.random.default_rng(EA.STEP_LATENCY_MS[1])   # fixed seed
    sample = list(rng.choice(pool, size=min(args.n, len(pool)), replace=False))

    verified_all = []
    first = True
    for did in sample:
        bundle, gen, verified, packet = run_pipeline(did, disputes, evidence)
        verified_all.append(verified)
        if first:
            print_example(did, bundle, gen, verified, packet)
            first = False

    cov, resolves, n = citation_coverage(verified_all)
    kept = sum(v["resolves"] and v["supported"] for vs in verified_all for v in vs)
    print("\n" + "=" * 72)
    print(f"METRICS over {len(sample)} disputes ({args.split} split)")
    print("=" * 72)
    print(f"  claims generated      : {n}")
    print(f"  citation coverage     : {cov*100:.1f}%  ({resolves}/{n} resolve)")
    print(f"  verifier kept         : {kept}/{n} ({kept/n*100:.1f}%)  "
          f"[{n-kept} stripped as unsupported]")

    print("\n  hallucination under stress (delete a cited artifact, regenerate):")
    stress = [stress_one(did, disputes, evidence, rng) for did in sample]
    stress = [s for s in stress if s]
    if stress:
        pre = np.mean([s["asserted_pre"] for s in stress])
        post = np.mean([s["asserted_post"] for s in stress])
        print(f"    stress cases          : {len(stress)}")
        print(f"    deleted fact asserted (generator, pre-verify) : {pre*100:.1f}%")
        print(f"    deleted fact asserted (packet,   post-verify) : {post*100:.1f}%")
        print(f"    -> verifier removed {(pre-post)*100:.1f} points of asserted-but-"
              f"unsupported facts")
    else:
        print("    no eligible stress cases in sample")

    print("\n  adversarial verifier check (fabricated claims that contradict artifacts):")
    adversarial_check(disputes, evidence, preds)


if __name__ == "__main__":
    main()
