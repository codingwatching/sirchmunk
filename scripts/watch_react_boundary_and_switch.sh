#!/usr/bin/env bash
# Boundary watcher: wait for react G_250 completion, then stop the v3 pipeline
# (which still runs the pre-tuning LENS config) and relaunch a 3-baseline
# dynamic continuation. LENS results are archived for后续调优对比 and removed
# from the runs dir so the post-tuning backfill (--skip-existing) re-runs LENS.
#
# The stale-index arm is intentionally OFF in this continuation: skip-existing
# loads cached fresh results without prepare(), so a stale arm launched from
# unprepared baseline instances would silently rebuild on the current corpus
# and corrupt the staleness measurement. Staleness is computed once in the
# final backfill pass instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REACT_FILE="benchmarks/hotpotqa/output/dynamic_eval/runs/G_250_D_250/baselines/baseline_react.jsonl"
BACKUP_DIR="benchmarks/hotpotqa/output/lens_pre_tuning_backup"
LOG="benchmarks/hotpotqa/output/logs/watcher_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > benchmarks/hotpotqa/output/logs/latest_watcher_log.txt

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

log "watcher started; waiting for react G_250 to reach 250"
while true; do
  n=$(wc -l < "$REACT_FILE" 2>/dev/null | tr -d ' ' || echo 0)
  log "react G_250 progress: ${n:-0}/250"
  if [ "${n:-0}" -ge 250 ]; then
    break
  fi
  sleep 300
done

log "react G_250 complete; stopping v3 pipeline"
pkill -TERM -f 'formal_run_v3' 2>/dev/null || true
sleep 3
pkill -TERM -f 'run_dynamic_evaluation.py' 2>/dev/null || true
sleep 8
pkill -KILL -f 'run_dynamic_evaluation.py' 2>/dev/null || true

log "archiving pre-tuning LENS artifacts to $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
for stage in G_125_D_125 G_250_D_250 G_500_D_500; do
  src="benchmarks/hotpotqa/output/dynamic_eval/runs/$stage/baselines/baseline_ablation_lens_full.jsonl"
  if [ -f "$src" ]; then
    mv "$src" "$BACKUP_DIR/${stage}_baseline_ablation_lens_full.jsonl"
    log "archived $src"
  fi
done

log "relaunching 3-baseline dynamic continuation (skip-existing, no stale arm)"
RELAUNCH_LOG="benchmarks/hotpotqa/output/logs/formal_run_v4_baselines_$(date +%Y%m%d_%H%M%S).log"
echo "$RELAUNCH_LOG" > benchmarks/hotpotqa/output/logs/latest_formal_log.txt
nohup python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --output-dir benchmarks/hotpotqa/output/dynamic_eval \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize copy \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines bm25_rag,hybrid_rag,react \
  --baseline-max-files 50000 \
  --skip-existing \
  >> "$RELAUNCH_LOG" 2>&1 &
log "relaunched (pid $!), log: $RELAUNCH_LOG"
log "watcher done"
