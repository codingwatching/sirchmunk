#!/bin/bash
# Paper experiment queue (2026-08-12): wait for the running G_500 dynamic run,
# then execute the remaining paper experiments sequentially to avoid API
# concurrency contention. Task list mirrors the tables in
# temp/papers/overleaf_version/lens_submission/aaai2027-lens-arxiv-v1.tex:
#   T1  dynamic nested stages 125,250      -> tab:evidence_recall + lifecycle tables
#   T2  stale-index arm (D_125 -> D_250)   -> tab:staleness
#   T3  fullwiki ReAct + closed_book (n=150, dev150 ids) -> tab:fullwiki / tab:scale fullwiki column + no-retrieval reference
#   T4  fullwiki ablations (n=150)         -> tab:ablation (lens_no_prior, lens_no_seq)
#   T5  controlled D_150 (LENS + ReAct)    -> tab:scale controlled column (sample-context corpus; owner-approved)
#   T6  judge semantic-acc rescore         -> auxiliary judge acc for dynamic stages (paper auxiliary metric)
# LENS fullwiki n=150 with the latest code already exists (run hotpotqa_20260811_234139).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="benchmarks/hotpotqa/output/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/paper_queue_${TS}.log"
echo "$LOG" > "$LOGDIR/latest_paper_queue_log.txt"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

STATUS_FILE="$LOGDIR/paper_queue_${TS}_status.tsv"
printf 'task\texit_code\tstarted\tfinished\n' > "$STATUS_FILE"

run_task() {
  local name="$1"; shift
  local task_log="$LOGDIR/paper_queue_${TS}_${name}.log"
  local started finished rc
  started="$(date '+%F %T')"
  log "START ${name}  log=${task_log}"
  "$@" > "$task_log" 2>&1
  rc=$?
  finished="$(date '+%F %T')"
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$started" "$finished" >> "$STATUS_FILE"
  log "END   ${name}  exit=${rc}"
  return 0  # continue queue even if one task fails; failures recorded in status file
}

# ---- Phase 0: wait for the in-flight G_500 dynamic run to finish -----------
log "queue started; waiting for dynamic_eval_g500_v1 processes to exit"
while true; do
  running=$(pgrep -f "dynamic_eval_g500_v1" | wc -l | tr -d ' ')
  if [ "$running" -eq 0 ]; then
    break
  fi
  n=$(wc -l < benchmarks/hotpotqa/output/dynamic_eval_g500_v1/runs/G_500_D_500/baselines/baseline_ablation_lens_full.jsonl 2>/dev/null | tr -d ' ' || echo 0)
  log "G_500 still running (procs=$running, lens_full=${n:-0}/500)"
  sleep 300
done
log "G_500 processes exited; starting queued tasks"

# ---- T1: nested dynamic stages 125,250 (same parent golden set) ------------
run_task t1_dynamic_125_250 \
  python benchmarks/run_dynamic_evaluation.py \
    --benchmark hotpotqa \
    --env benchmarks/hotpotqa/.env.hotpotqa.tuning \
    --output-dir benchmarks/hotpotqa/output/dynamic_eval_g500_v1 \
    --golden-n 500 --seed 42 --stages 125,250 \
    --materialize copy \
    --run-baselines \
    --baselines lens_full,bm25_rag,hybrid_rag,react,closed_book \
    --skip-existing

# ---- T2: stale-index arm (index on D_125, evaluate on D_250) ---------------
run_task t2_stale_index_arm \
  python benchmarks/run_dynamic_evaluation.py \
    --benchmark hotpotqa \
    --env benchmarks/hotpotqa/.env.hotpotqa.tuning \
    --output-dir benchmarks/hotpotqa/output/dynamic_eval_g500_v1 \
    --golden-n 500 --seed 42 --stages 125,250 \
    --materialize copy \
    --run-baselines \
    --baselines bm25_rag,hybrid_rag,react,lens_full \
    --stale-index-arm \
    --skip-existing

# ---- T3: fullwiki ReAct + closed_book (n=150, dev150 fixed ids) ------------
run_task t3_fullwiki_react \
  python benchmarks/run_evaluation.py \
    --benchmark hotpotqa \
    --env benchmarks/hotpotqa/.env.hotpotqa.fullwiki_experiment \
    --output-dir benchmarks/hotpotqa/output/fullwiki_v2/react \
    --sampling-method fixed_ids \
    --sample-ids-file benchmarks/hotpotqa/dev150_sample_ids.json \
    --baselines react,closed_book

# ---- T4: fullwiki ablations (n=150, dev150 fixed ids) ----------------------
run_task t4_fullwiki_ablations \
  python benchmarks/run_evaluation.py \
    --benchmark hotpotqa \
    --env benchmarks/hotpotqa/.env.hotpotqa.fullwiki_experiment \
    --output-dir benchmarks/hotpotqa/output/fullwiki_v2/ablations \
    --sampling-method fixed_ids \
    --sample-ids-file benchmarks/hotpotqa/dev150_sample_ids.json \
    --baselines lens_no_prior,lens_no_seq

# ---- T5: controlled D_150 arm (LENS + ReAct, sample-context corpus) --------
# Corpus mode sample-context approved by owner for tab:scale controlled column.
HOTPOT_CONTEXT_CORPUS_MODE=sample \
run_task t5_d150_controlled \
  python benchmarks/run_evaluation.py \
    --benchmark hotpotqa \
    --env benchmarks/hotpotqa/.env.hotpotqa.tuning \
    --output-dir benchmarks/hotpotqa/output/d150_controlled_v2 \
    --sampling-method fixed_ids \
    --sample-ids-file benchmarks/hotpotqa/dev150_sample_ids.json \
    --baselines lens_full,react

# ---- T6: judge semantic-acc rescore over dynamic stages (auxiliary metric) --
# Zero retrieval re-run: EM hits take the lexical fast path; only non-EM
# answers go to the isolated judge model (HOTPOT_JUDGE_MODEL_NAME).
run_task t6_judge_rescore \
  python benchmarks/hotpotqa/rescore_judge.py \
    --runs benchmarks/hotpotqa/output/dynamic_eval_g500_v1/runs \
    --env benchmarks/hotpotqa/.env.hotpotqa.base \
    --concurrency 5

log "queue finished; status:"
cat "$STATUS_FILE" >> "$LOG"
