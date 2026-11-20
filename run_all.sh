#!/usr/bin/env bash
# Full pipeline: teacher data -> shortcut distillation -> PartiPrompts evaluation.
# Setup first: bash setup_env.sh install && bash setup_env.sh download   (same ROOT for both)
# Uses all visible GPUs (NGPU); on >=60 GB GPUs gradient checkpointing is off (scripts/gpu_defaults.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=${ROOT:-$HOME/shortcut_work}
ENV_FILE=${ENV_FILE:-$ROOT/shortcut_env.sh}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
# shellcheck disable=SC1091
source scripts/gpu_defaults.sh
# models / PartiPrompts are pre-downloaded by `setup_env.sh download`; without this every
# from_pretrained() call first pings huggingface.co, which is slow or unreachable on some networks
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

RES=${RES:-512}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
DATA_ROOT=${DATA_ROOT:-$ROOT}   # data, runs and evaluation outputs (large: keep on a big disk)
DATA=${DATA:-$DATA_ROOT/data/sd35m_${RES}}
RUN=${RUN:-$DATA_ROOT/runs/shortcut_${RES}}
EVAL=${EVAL:-$DATA_ROOT/eval/${RES}}
# same number of training samples for any NGPU (per-GPU batch 8, global batch 8*NGPU)
STEPS=${STEPS:-$((6000 / NGPU))}
mkdir -p logs

# 1) prompts (already committed as data/prompts.txt; disjoint from PartiPrompts)
[ -f data/prompts.txt ] || python scripts/prepare_prompts.py --out data/prompts.txt --total 4000

# 2) teacher data (SD3.5M, 40 steps, CFG 4.5): one worker per GPU, shards split between them
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_data.py --prompts data/prompts.txt \
      --out "$DATA" --height "$RES" --width "$RES" --batch_size "$GEN_BS" \
      --num_workers "$NGPU" --worker_id "$i" > "logs/gen_$i.log" 2>&1 &
  pids+=($!)
done
echo "generating data on $NGPU GPU(s) -> $DATA ; progress: tail -f logs/gen_*.log"
for p in "${pids[@]}"; do wait "$p"; done

# 3) shortcut distillation (LoRA r64 + zero-init d-embedder)
accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/train.py \
    --data_dir "$DATA" --output_dir "$RUN" $GC_FLAG \
    --batch_size 8 --max_steps "$STEPS" 2>&1 | tee logs/train.log

# 4) PartiPrompts (first 100): teacher 40 steps vs naive 4 steps vs student 1/2/4/8
python scripts/evaluate.py --student "$RUN/final/student_ema" --out "$EVAL" \
    --height "$RES" --width "$RES" --batch_size 8 --pickscore 2>&1 | tee logs/eval.log
echo "done: metrics in $EVAL/metrics.md, grids in $EVAL/grid_*.png"
