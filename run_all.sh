#!/usr/bin/env bash
# Full pipeline: teacher data -> shortcut distillation -> PartiPrompts evaluation.
# Setup first: bash setup_env.sh install && bash setup_env.sh download   (same ROOT for both)
# Uses all visible GPUs (NGPU); on >=60 GB GPUs gradient checkpointing is off (scripts/gpu_defaults.sh).
#
# Configuration (environment variables, defaults = run 1 of the README):
#   RES=512                 training / data resolution (square)
#   PROMPTS=data/prompts.txt
#   RUN_NAME=shortcut_$RES  name of the run (runs/$RUN_NAME, eval/$RUN_NAME/<WxH>)
#   BATCH=8 ACCUM=1         per-GPU batch and gradient accumulation (effective batch BATCH*ACCUM*NGPU)
#   STEPS=6000/NGPU         optimizer steps
#   TRAIN_ARGS=""           extra arguments for scripts/train.py (e.g. "--cfg_scale 3.0")
#   EVAL_SIZES="RESxRES"    space-separated WxH list for evaluation (e.g. "1024x1024 832x480")
#   EVAL_STEPS=1,2,4,8      student step counts to evaluate
# Example (run 2):
#   RES=1024 PROMPTS=data/prompts_v2.txt RUN_NAME=shortcut_1024_v2 DATA=$ROOT/data/sd35m_1024_v2 \
#   BATCH=4 ACCUM=2 GEN_BS=8 \
#   TRAIN_ARGS="--cfg_scale 3.0 --bootstrap_fraction 0.5" EVAL_SIZES="1024x1024 832x480" bash run_all.sh
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
PROMPTS=${PROMPTS:-data/prompts.txt}
RUN_NAME=${RUN_NAME:-shortcut_${RES}}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-1}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_SIZES=${EVAL_SIZES:-${RES}x${RES}}
EVAL_STEPS=${EVAL_STEPS:-1,2,4,8}
DATA_ROOT=${DATA_ROOT:-$ROOT}   # data, runs and evaluation outputs (large: keep on a big disk)
DATA=${DATA:-$DATA_ROOT/data/sd35m_${RES}}
RUN=${RUN:-$DATA_ROOT/runs/$RUN_NAME}
EVAL=${EVAL:-$DATA_ROOT/eval/$RUN_NAME}
# same number of training samples for any NGPU
STEPS=${STEPS:-$((6000 / NGPU))}
mkdir -p logs
echo "run $RUN_NAME: RES=$RES PROMPTS=$PROMPTS BATCH=$BATCH ACCUM=$ACCUM STEPS=$STEPS TRAIN_ARGS='$TRAIN_ARGS'"

# 1) prompts (committed in data/; disjoint from PartiPrompts)
[ -f "$PROMPTS" ] || python scripts/prepare_prompts.py --out "$PROMPTS" --total 4000

# 2) teacher data (SD3.5M, 40 steps, CFG 4.5): one worker per GPU, shards split between them
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_data.py --prompts "$PROMPTS" \
      --out "$DATA" --height "$RES" --width "$RES" --batch_size "$GEN_BS" \
      --num_workers "$NGPU" --worker_id "$i" > "logs/gen_$i.log" 2>&1 &
  pids+=($!)
done
echo "generating data on $NGPU GPU(s) -> $DATA ; progress: tail -f logs/gen_*.log"
for p in "${pids[@]}"; do wait "$p"; done

# 3) shortcut distillation (LoRA r64 + zero-init d-embedder)
# shellcheck disable=SC2086
accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/train.py \
    --data_dir "$DATA" --output_dir "$RUN" $GC_FLAG \
    --batch_size "$BATCH" --grad_accum "$ACCUM" --max_steps "$STEPS" $TRAIN_ARGS 2>&1 | tee logs/train.log

# 4) PartiPrompts (first 100) + task prompts: teacher 40 steps vs naive 4 steps vs student
for size in $EVAL_SIZES; do
  w=${size%x*}; h=${size#*x}
  python scripts/evaluate.py --student "$RUN/final/student_ema" --out "$EVAL/$size" \
      --height "$h" --width "$w" --steps "$EVAL_STEPS" --batch_size 8 --pickscore \
      2>&1 | tee "logs/eval_$size.log"
done
echo "done: metrics in $EVAL/*/metrics.md, grids in $EVAL/*/grid_*.png"
