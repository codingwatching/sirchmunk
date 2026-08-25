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
#   TRIAL=1 bash scripts/run_paper_experiments.sh <stage>   # small trial (n=10)
#   stages: smoke | sync | sampling | validate | assets | dynamic | main | report | status | ablation | all
#
# Notes:
#   - Frozen/paper-facing stages all use benchmarks/hotpotqa/.env.hotpotqa.frozen.
#   - Every frozen stage shares the same sample IDs file (Gate 2 pairing).
#   - Sampling is gated on raw-corpus synchronization: `create` resolves every
#     supporting-fact article against the enwiki dump and records the dump
#     fingerprint (wiki_corpus_fingerprint) in the sample-ids metadata.
#   - Dynamic baselines include lens_full: the dynamic task does not run LENS
#     implicitly, and the stale-index arm needs LENS as the index-free control.
#   - TRIAL=1 exercises every link with n=10, stages 5,10, and a separate
#     output root (output/trial) so formal artifacts are never touched.
#   - "all" runs the full frozen path in order; smoke is excluded (exploration only).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_EXPLORATION="benchmarks/hotpotqa/.env.hotpotqa.exploration"
ENV_FROZEN="benchmarks/hotpotqa/.env.hotpotqa.frozen"

TRIAL="${TRIAL:-0}"
if [[ "$TRIAL" == "1" ]]; then
  OUTPUT_DIR="benchmarks/hotpotqa/output/trial"
  TARGET_N=10
  DYNAMIC_STAGES="5,10"
  STALENESS_MAX=10
  REPORT_TITLE="HotpotQA Fullwiki ResearchOps Trial Report"
else
  OUTPUT_DIR="benchmarks/hotpotqa/output"
  TARGET_N=500
  DYNAMIC_STAGES="125,250,500"
  STALENESS_MAX=200
  REPORT_TITLE="HotpotQA Fullwiki ResearchOps Report"
fi

DYNAMIC_BASELINES="bm25_rag,hybrid_rag,react,lens_full"
SAMPLING_DIR="$OUTPUT_DIR/main/sampling"
SAMPLE_IDS="$SAMPLING_DIR/sampling_stratified_42_${TARGET_N}_sample_ids.json"
SAMPLE_MANIFEST="$SAMPLING_DIR/sampling_stratified_42_${TARGET_N}_manifest.json"

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
    log "Stage 1: create frozen stratified sample IDs (n=$TARGET_N, seed=42, corpus-sync gated)"
    python benchmarks/run_sampling.py create \
      --benchmark hotpotqa \
      --env "$ENV_FROZEN" \
      --method stratified \
      --target-n "$TARGET_N" \
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
  log "Stage 3: dynamic G_n/D_n evaluation with stale-index arm (baselines: $DYNAMIC_BASELINES)"
  # copy materialization is required for retrieval correctness: ripgrep-based
  # search (LENS rga channel, ReAct keyword search) does not follow symlinks,
  # so a symlink snapshot silently blinds every grep-backed system.
  python benchmarks/run_benchmark.py dynamic \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --output-dir "$OUTPUT_DIR/dynamic_eval" \
    --golden-n "$TARGET_N" \
    --seed 42 \
    --stages "$DYNAMIC_STAGES" \
    --strata type,supporting_fact_bucket \
    --materialize copy \
    --background-ratio 3.0 \
    --background-seed 42 \
    --run-baselines \
    --baselines "$DYNAMIC_BASELINES" \
    --baseline-max-files 50000 \
    --stale-index-arm \
    --staleness-max-samples "$STALENESS_MAX"
}

# -----------------------------------------------------------------------------
# Stage 4: frozen main experiment (Sirchmunk/LENS + core paper-main baselines).
# Cold cache, strict Gate 0-5, shared frozen sample IDs.
# -----------------------------------------------------------------------------
stage_main() {
  log "Stage 4: frozen main experiment (Sirchmunk + bm25_rag,hybrid_rag,react)"
  # cold cache must actually clear caches, otherwise Gate 5 cache_policy errors
  # ("cold cache requested but allow_clear=False"). Clearing is restricted to
  # cache dirs under work_path by CacheManager.
  CACHE_ALLOW_CLEAR=true python benchmarks/run_benchmark.py main \
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
  if [[ -z "$run_id" && -f "$OUTPUT_DIR/main/main_summary.json" ]]; then
    # main_summary.json is the authoritative run-status artifact
    run_id="$(python -c "import json;print(json.load(open('$OUTPUT_DIR/main/main_summary.json')).get('control_run_id',''))")"
  fi
  if [[ -z "$run_id" ]]; then
    echo "No main run recorded in $OUTPUT_DIR/main/main_summary.json; run stage 'main' first." >&2
    exit 1
  fi
  # Sirchmunk run artifacts live under HOTPOT_OUTPUT_DIR (frozen profile:
  # output/frozen/runs), not under the facade --output-dir main/runs.
  local run_dir=""
  for candidate in \
    "$OUTPUT_DIR/main/runs/$run_id" \
    "benchmarks/hotpotqa/output/frozen/runs/$run_id"; do
    if [[ -d "$candidate" ]]; then run_dir="$candidate"; break; fi
  done
  if [[ -z "$run_dir" ]]; then
    echo "Run directory for '$run_id' not found under main/runs or frozen/runs." >&2
    exit 1
  fi
  log "Stage 5: regenerate report for run $run_id ($run_dir)"
  python benchmarks/run_benchmark.py report \
    --run-dir "$run_dir" \
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
# Stage 6: frozen mechanism ablation (paper tab:ablation).
# Runs the three semantic LENS profiles (lens_full / lens_no_prior /
# lens_no_seq) through the same evaluation suite and frozen sample IDs as the
# main experiment. This intentionally avoids the env-grid ablation matrix:
# several of its axes are not wired to the search pipeline, so its variants do
# not correspond to the paper's prior/sequential mechanism design.
# -----------------------------------------------------------------------------
stage_ablation() {
  log "Stage 6: frozen LENS mechanism ablation (lens_full,lens_no_prior,lens_no_seq)"
  CACHE_ALLOW_CLEAR=true python benchmarks/run_evaluation.py \
    --benchmark hotpotqa \
    --env "$ENV_FROZEN" \
    --sample-ids-file "$SAMPLE_IDS" \
    --baselines lens_full,lens_no_prior,lens_no_seq \
    --ours-name "LENS" \
    --output-dir "$OUTPUT_DIR/ablation" \
    --baseline-sample-timeout 0 \
    --baseline-max-runtime 0 \
    --baseline-max-total-tokens 0 \
    --baseline-max-api-cost-usd 0 \
    --baseline-max-disk-bytes 0 \
    --baseline-min-free-disk-bytes 0 \
    --context-corpus-provenance wiki \
    --context-corpus-risk raw_wiki \
    --generate-report
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
