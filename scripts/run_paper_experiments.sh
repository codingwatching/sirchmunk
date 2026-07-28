#!/usr/bin/env bash
# =============================================================================
# LENS paper (AAAI2027) full formal experiment pipeline.
#
# Source of truth: benchmarks/README.md (ResearchOps guide).
# Paper artifacts covered:
#   tab:main_results   -> stage dynamic (G_n/D_n main tables) + stage main
#   tab:lifecycle      -> stage assets + stage dynamic (lifecycle_main, update_readiness)
#   fig:budget_tradeoff-> stage dynamic (budget_quality tables)
#   tab:ablation       -> stage ablation (lens_full / lens_no_prior / lens_no_seq)
#   stale-index claim  -> stage dynamic (--stale-index-arm)
#
# Usage:
#   bash scripts/run_paper_experiments.sh <stage>
#   stages: smoke | sync | sampling | validate | assets | dynamic | main | report | status | ablation | all
#
# Notes:
#   - Frozen/paper-facing stages all use benchmarks/hotpotqa/.env.hotpotqa.frozen.
#   - Every frozen stage shares the same sample IDs file (Gate 2 pairing).
#   - Sampling is gated on raw-corpus synchronization: `create` resolves every
#     supporting-fact article against the enwiki dump and records the dump
#     fingerprint (wiki_corpus_fingerprint) in the sample-ids metadata.
#   - "all" runs the full frozen path in order; smoke is excluded (exploration only).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_EXPLORATION="benchmarks/hotpotqa/.env.hotpotqa.exploration"
ENV_FROZEN="benchmarks/hotpotqa/.env.hotpotqa.frozen"
OUTPUT_DIR="benchmarks/hotpotqa/output"
SAMPLING_DIR="$OUTPUT_DIR/main/sampling"
SAMPLE_IDS="$SAMPLING_DIR/sampling_stratified_42_500_sample_ids.json"
SAMPLE_MANIFEST="$SAMPLING_DIR/sampling_stratified_42_500_manifest.json"
REPORT_TITLE="HotpotQA Fullwiki ResearchOps Report"

log() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*"; }

# -----------------------------------------------------------------------------
# Stage 0 (optional, NOT paper evidence): exploration smoke over sample contexts.
# Verifies env, data path, retrieval, judging, artifact writing, report smoke.
# -----------------------------------------------------------------------------
stage_smoke() {
  log "Stage 0: mock/smoke exploration (not paper evidence)"
  python benchmarks/run_benchmark.py smoke-tune \
    --benchmark hotpotqa \
    --env "$ENV_EXPLORATION" \
    --limit 20 \
    --seed 42 \
    --max-iter 1 \
    --context-corpus-mode sample \
    --run-evaluation \
    --baselines bm25,naive_rag,bm25_rag,hybrid_rag,react \
    --baseline-sample-timeout 0 \
    --baseline-max-runtime 0 \
    --generate-evaluation-report
}

# -----------------------------------------------------------------------------
# Stage 1a: raw-corpus synchronization gate. Blocks freezing when any
# supporting-fact article is missing from the raw enwiki dump.
# First run scans the dump once (~30s) and caches a fingerprinted title index.
# -----------------------------------------------------------------------------
stage_sync() {
  log "Stage 1a: raw-corpus synchronization check"
  python benchmarks/run_sampling.py check-corpus-sync \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN"
}

# -----------------------------------------------------------------------------
# Stage 1: freeze the sample IDs (stratified n=500, seed=42).
# Re-freezes when the existing artifact predates the corpus-sync gate
# (missing wiki_corpus_fingerprint in sample-ids metadata).
# -----------------------------------------------------------------------------
sample_ids_fingerprinted() {
  [[ -f "$SAMPLE_IDS" ]] || return 1
  python - "$SAMPLE_IDS" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
fp = (data.get("metadata") or {}).get("wiki_corpus_fingerprint")
sys.exit(0 if fp else 1)
PY
}

stage_sampling() {
  stage_sync
  if sample_ids_fingerprinted; then
    log "Stage 1: fingerprinted frozen sample IDs already exist, skipping create: $SAMPLE_IDS"
  else
    log "Stage 1: create frozen stratified sample IDs (n=500, seed=42, corpus-sync gated)"
    python benchmarks/run_sampling.py create \
      --benchmark hotpotqa \
      --env "$ENV_FROZEN" \
      --method stratified \
      --target-n 500 \
      --seed 42 \
      --strata type,supporting_fact_bucket \
      --allocation proportional \
      --min-per-stratum 1 \
      --expected-population-size 7405 \
      --output-dir "$SAMPLING_DIR" \
      --force
  fi
  stage_validate
}

# -----------------------------------------------------------------------------
# Stage 1b: validate manifest/checksum before any expensive system runs.
# -----------------------------------------------------------------------------
stage_validate() {
  log "Stage 1b: validate sampling manifest and checksum"
  python benchmarks/run_sampling.py validate \
    --manifest "$SAMPLE_MANIFEST"
}

# -----------------------------------------------------------------------------
# Stage 2: frozen baseline asset preflight (lifecycle evidence -> tab:lifecycle).
# --limit 20 is a preflight sample for path inference only; final evaluation
# size is governed by the frozen sample IDs file (n=500).
# -----------------------------------------------------------------------------
stage_assets() {
  log "Stage 2: build frozen baseline assets (lifecycle evidence)"
  python benchmarks/run_benchmark.py assets \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --output-dir "$OUTPUT_DIR" \
    --methods bm25,bm25_rag,naive_rag,react \
    --limit 20 \
    --seed 42 \
    --corpus-scale fullwiki \
    --build-timeout 86400 \
    --max-ram-bytes 0 \
    --max-disk-bytes 500000000000 \
    --max-llm-tokens 0 \
    --max-api-cost-usd 0 \
    --retry-count 0 \
    --bm25-max-files 20000 \
    --naive-rag-max-files 5000 \
    --stage frozen \
    --strict

  log "Stage 2b: inspect asset registry"
  python benchmarks/run_benchmark.py status \
    --asset-registry "$OUTPUT_DIR/assets/asset_registry.jsonl" \
    --benchmark hotpotqa \
    --methods bm25,bm25_rag,naive_rag,react \
    --stage frozen \
    --reusable-only
}

# -----------------------------------------------------------------------------
# Stage 3: dynamic G_n/D_n evaluation (paper main dynamic protocol).
# Produces: dynamic_main_results, lifecycle_main, budget_quality,
# update_readiness, snapshot_audit, stale_index_quality.
# Nested stages 125/250/500 with the stale-index arm enabled.
# -----------------------------------------------------------------------------
stage_dynamic() {
  log "Stage 3: dynamic G_n/D_n evaluation with stale-index arm"
  python benchmarks/run_benchmark.py dynamic \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --golden-n 500 \
    --seed 42 \
    --stages 125,250,500 \
    --strata type,supporting_fact_bucket \
    --materialize symlink \
    --background-ratio 3.0 \
    --background-seed 42 \
    --run-baselines \
    --baselines bm25_rag,hybrid_rag,react \
    --stale-index-arm \
    --staleness-max-samples 200
}

# -----------------------------------------------------------------------------
# Stage 4: frozen main experiment (Sirchmunk/LENS + core paper-main baselines).
# Cold cache, strict Gate 0-5, shared frozen sample IDs.
# -----------------------------------------------------------------------------
stage_main() {
  log "Stage 4: frozen main experiment (Sirchmunk + bm25_rag,hybrid_rag,react)"
  python benchmarks/run_benchmark.py main \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --output-dir "$OUTPUT_DIR" \
    --run-sirchmunk \
    --sample-ids-file "$SAMPLE_IDS" \
    --expected-population-size 7405 \
    --baselines bm25_rag,hybrid_rag,react \
    --cache-mode cold \
    --baseline-sample-timeout 0 \
    --baseline-max-runtime 0 \
    --baseline-max-total-tokens 0 \
    --baseline-max-api-cost-usd 0 \
    --baseline-max-disk-bytes 0 \
    --baseline-min-free-disk-bytes 0 \
    --generate-report \
    --title "$REPORT_TITLE" \
    --strict
}

# -----------------------------------------------------------------------------
# Stage 5: regenerate report from frozen artifacts and inspect gate status.
# RUN_ID must point to the frozen main run directory.
# -----------------------------------------------------------------------------
stage_report() {
  local run_id="${RUN_ID:-}"
  if [[ -z "$run_id" ]]; then
    run_id="$(ls -1t "$OUTPUT_DIR/main/runs" 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$run_id" ]]; then
    echo "No main run found under $OUTPUT_DIR/main/runs; run stage 'main' first." >&2
    exit 1
  fi
  log "Stage 5: regenerate report for run $run_id"
  python benchmarks/run_benchmark.py report \
    --run-dir "$OUTPUT_DIR/main/runs/$run_id" \
    --table-json "$OUTPUT_DIR/main/evaluation/paper_table.json" \
    --output-dir "$OUTPUT_DIR/main/report" \
    --title "$REPORT_TITLE" \
    --stage frozen \
    --strict
  stage_status
}

stage_status() {
  log "Stage 5b: inspect frozen run summary"
  python benchmarks/run_benchmark.py status \
    --summary "$OUTPUT_DIR/main/main_summary.json"
}

# -----------------------------------------------------------------------------
# Stage 6: frozen mechanism ablation (tab:ablation).
# Core variants: lens_full / lens_no_prior / lens_no_seq (lens_no_reuse stays
# appendix-only per paper design).
# -----------------------------------------------------------------------------
stage_ablation() {
  log "Stage 6: frozen LENS mechanism ablation"
  python benchmarks/run_benchmark.py ablation \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --sample-ids-file "$SAMPLE_IDS" \
    --cache-mode cold \
    --max-combinations 16 \
    --run \
    --max-concurrent 1 \
    --replace
}

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}"
  exit 1
}

main() {
  local stage="${1:-}"
  case "$stage" in
    smoke)    stage_smoke ;;
    sync)     stage_sync ;;
    sampling) stage_sampling ;;
    validate) stage_validate ;;
    assets)   stage_assets ;;
    dynamic)  stage_dynamic ;;
    main)     stage_main ;;
    report)   stage_report ;;
    status)   stage_status ;;
    ablation) stage_ablation ;;
    all)
      stage_sampling
      stage_assets
      stage_dynamic
      stage_main
      stage_report
      stage_ablation
      ;;
    *) usage ;;
  esac
  log "Done: stage '$stage' finished"
}

main "$@"
