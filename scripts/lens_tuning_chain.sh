#!/usr/bin/env bash
# Tuning iteration chain: when the dev-50 reference run (pre-T2 code, module
# already loaded in its process) finishes, archive its results and launch the
# iter1 run in a fresh process so the T2 synthesis changes take effect.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REF_JSONL="benchmarks/hotpotqa/output/lens_tuning/dynamic_eval/runs/G_50_D_50/baselines/baseline_ablation_lens_full.jsonl"
ARCHIVE="benchmarks/hotpotqa/output/lens_tuning/ref_topk10_no_t2"
LOG="benchmarks/hotpotqa/output/logs/lens_tuning_chain_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > benchmarks/hotpotqa/output/logs/latest_tuning_chain_log.txt
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

log "waiting for dev-50 reference run to finish"
while true; do
  n=$(wc -l < "$REF_JSONL" 2>/dev/null | tr -d ' ' || echo 0)
  running=$(pgrep -f "lens_tuning/dynamic_eval" | wc -l | tr -d ' ')
  log "reference progress: ${n:-0}/50 (procs=$running)"
  if [ "${n:-0}" -ge 50 ] && [ "$running" -eq 0 ]; then
    break
  fi
  sleep 180
done

log "archiving reference results to $ARCHIVE"
mkdir -p "$ARCHIVE"
cp -p "$REF_JSONL" "$ARCHIVE/" 2>/dev/null || true
cp -pr benchmarks/hotpotqa/output/lens_tuning/dynamic_eval/tables "$ARCHIVE/tables" 2>/dev/null || true

log "launching iter1 (T2: trim-fix + verbatim contract + gated refusal fallback)"
ITER_LOG="benchmarks/hotpotqa/output/logs/lens_tuning_iter1_$(date +%Y%m%d_%H%M%S).log"
echo "$ITER_LOG" > benchmarks/hotpotqa/output/logs/latest_tuning_log.txt
nohup python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.tuning \
  --output-dir benchmarks/hotpotqa/output/lens_tuning/iter1/dynamic_eval \
  --golden-n 50 \
  --seed 105 \
  --stages 50 \
  --strata type,supporting_fact_bucket \
  --materialize copy \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines lens_full \
  --baseline-max-files 50000 \
  > "$ITER_LOG" 2>&1 &
log "iter1 launched (pid $!), log: $ITER_LOG"
log "chain done"
