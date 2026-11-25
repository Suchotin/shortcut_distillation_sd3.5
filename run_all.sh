#!/usr/bin/env bash
# Prompts -> teacher data -> shortcut distillation -> evaluation on PartiPrompts.
# Run `bash setup_env.sh install && bash setup_env.sh download` first (same ROOT).
#
#   RES=1024  PROMPTS=data/prompts_v2.txt  RUN_NAME=shortcut_1024  STEPS=6000
#   BATCH=4 ACCUM=2 (per GPU)  TRAIN_ARGS=""  EVAL_SIZES="1024x1024 832x480"  DATA_ROOT=$ROOT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=${ROOT:-$HOME/shortcut_work}
# shellcheck disable=SC1091
[ -f "$ROOT/shortcut_env.sh" ] && source "$ROOT/shortcut_env.sh"
# shellcheck disable=SC1091
source scripts/gpu_defaults.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

RES=${RES:-1024}
PROMPTS=${PROMPTS:-data/prompts_v2.txt}
RUN_NAME=${RUN_NAME:-shortcut_$RES}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-2}
STEPS=${STEPS:-6000}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_SIZES=${EVAL_SIZES:-"${RES}x${RES} 832x480"}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
DATA_ROOT=${DATA_ROOT:-$ROOT}
DATA=${DATA:-$DATA_ROOT/data/sd35m_$RES}
RUN=$DATA_ROOT/runs/$RUN_NAME
EVAL=$DATA_ROOT/eval/$RUN_NAME
mkdir -p logs

[ -f "$PROMPTS" ] || python scripts/prepare_prompts.py --out "$PROMPTS" \
    --sources_out "${PROMPTS%.txt}_sources.jsonl"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_data.py --prompts "$PROMPTS" --out "$DATA" \
      --height "$RES" --width "$RES" --batch_size "$GEN_BS" \
      --num_workers "$NGPU" --worker_id "$i" > "logs/gen_$i.log" 2>&1 &
  pids+=($!)
done
echo "generating teacher data -> $DATA (tail -f logs/gen_*.log)"
for p in "${pids[@]}"; do wait "$p"; done

# shellcheck disable=SC2086
accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/train.py \
    --data_dir "$DATA" --output_dir "$RUN" --batch_size "$BATCH" --grad_accum "$ACCUM" \
    --max_steps "$STEPS" $GC_FLAG $TRAIN_ARGS 2>&1 | tee "logs/train_$RUN_NAME.log"

for size in $EVAL_SIZES; do
  python scripts/evaluate.py --student "$RUN/final/student_ema" --out "$EVAL/$size" \
      --width "${size%x*}" --height "${size#*x}" --batch_size 8 --pickscore \
      2>&1 | tee "logs/eval_${RUN_NAME}_$size.log"
done
echo "done: $EVAL/*/metrics.md"
