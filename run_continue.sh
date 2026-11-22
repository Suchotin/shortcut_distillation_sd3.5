#!/usr/bin/env bash
# Continue a finished run from its final checkpoint, optionally on extra teacher data
# (run 2b of the README: run 2 + 3,925 fresh 1024² images, 6,000 → 12,000 steps).
#
# Configuration (environment variables, defaults = run 2b):
#   BASE_RUN=$DATA_ROOT/runs/shortcut_1024_v2       run to continue (needs final/train_state.pt)
#   BASE_DATA=$DATA_ROOT/data/sd35m_1024_v2         its training data
#   NEW_PROMPTS=data/prompts_v3.txt                 extra prompts ("" = no extra data)
#   NEW_DATA=$DATA_ROOT/data/sd35m_1024_v3          where the extra teacher data goes
#   RES=1024 BATCH=4 ACCUM=2 GEN_BS=8               must match the base run
#   STEPS=12000                                     total optimizer steps, including the base run's
#   TRAIN_ARGS="--cfg_scale 3.0 --bootstrap_fraction 0.5"   must match the base run
#   RUN_NAME=shortcut_1024_v2b  EVAL_SIZES="1024x1024 832x480"  EVAL_STEPS=1,2,4,8
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=${ROOT:-$HOME/shortcut_work}
ENV_FILE=${ENV_FILE:-$ROOT/shortcut_env.sh}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
GEN_BS=${GEN_BS:-8}   # as in run 2 (noise does not depend on it)
# shellcheck disable=SC1091
source scripts/gpu_defaults.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

DATA_ROOT=${DATA_ROOT:-$ROOT}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
RES=${RES:-1024}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-2}
STEPS=${STEPS:-12000}
TRAIN_ARGS=${TRAIN_ARGS:-"--cfg_scale 3.0 --bootstrap_fraction 0.5"}
BASE_RUN=${BASE_RUN:-$DATA_ROOT/runs/shortcut_1024_v2}
BASE_DATA=${BASE_DATA:-$DATA_ROOT/data/sd35m_1024_v2}
NEW_PROMPTS=${NEW_PROMPTS-data/prompts_v3.txt}
NEW_DATA=${NEW_DATA:-$DATA_ROOT/data/sd35m_1024_v3}
RUN_NAME=${RUN_NAME:-shortcut_1024_v2b}
RUN=$DATA_ROOT/runs/$RUN_NAME
EVAL=$DATA_ROOT/eval/$RUN_NAME
EVAL_SIZES=${EVAL_SIZES:-${RES}x${RES} 832x480}
EVAL_STEPS=${EVAL_STEPS:-1,2,4,8}
[ -f "$BASE_RUN/final/train_state.pt" ] || { echo "no $BASE_RUN/final/train_state.pt"; exit 1; }
mkdir -p logs
echo "continue $BASE_RUN -> $RUN_NAME: STEPS=$STEPS NEW_PROMPTS='$NEW_PROMPTS' TRAIN_ARGS='$TRAIN_ARGS'"

DATA=$BASE_DATA
if [ -n "$NEW_PROMPTS" ]; then
  # 1) extra teacher data (same teacher settings, different noise seed)
  pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_data.py --prompts "$NEW_PROMPTS" \
        --out "$NEW_DATA" --height "$RES" --width "$RES" --batch_size "$GEN_BS" --seed 1 \
        --num_workers "$NGPU" --worker_id "$i" > "logs/gen_extra_$i.log" 2>&1 &
    pids+=($!)
  done
  echo "generating extra data on $NGPU GPU(s) -> $NEW_DATA ; progress: tail -f logs/gen_extra_*.log"
  for p in "${pids[@]}"; do wait "$p"; done

  # 2) one data directory with both shard sets (symlinks, consecutive shard numbers)
  DATA=${MIX_DATA:-${BASE_DATA}+$(basename "$NEW_DATA")}
  mkdir -p "$DATA" && find "$DATA" -maxdepth 1 -type l -delete   # only our own symlinks
  n=0
  for src in "$BASE_DATA" "$NEW_DATA"; do
    for f in "$src"/shard_*.safetensors; do
      tag=$(printf 'shard_%05d' "$n")
      ln -s "$f" "$DATA/$tag.safetensors"
      [ -f "${f%.safetensors}.json" ] && ln -s "${f%.safetensors}.json" "$DATA/$tag.json"
      n=$((n + 1))
    done
  done
  # null-prompt embedding, task showcase and teacher settings are identical for both sets
  cp "$BASE_DATA/null_embeds.safetensors" "$BASE_DATA/showcase.safetensors" "$DATA/"
  [ -f "$BASE_DATA/showcase.json" ] && cp "$BASE_DATA/showcase.json" "$DATA/"
  python - "$BASE_DATA/meta.json" "$NEW_DATA/meta.json" "$DATA/meta.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("model", "height", "width", "steps", "cfg", "shift", "latent_shape", "max_sequence_length"):
    assert a.get(k) == b.get(k), f"teacher setting differs between data sets: {k}"
a["n"] = a["n"] + b["n"]
json.dump(a, open(sys.argv[3], "w"), indent=2)
EOF
  echo "training data: $n shards in $DATA"
fi

# 3) continue training (optimizer, EMA, LR schedule and step counter come from the checkpoint;
#    a new seed gives fresh noise / time / level draws)
# shellcheck disable=SC2086
accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/train.py \
    --data_dir "$DATA" --output_dir "$RUN" --resume "$BASE_RUN/final" --seed 1 $GC_FLAG \
    --batch_size "$BATCH" --grad_accum "$ACCUM" --max_steps "$STEPS" $TRAIN_ARGS 2>&1 \
    | tee logs/train_continue.log

# 4) evaluation, as in run_all.sh
for size in $EVAL_SIZES; do
  w=${size%x*}; h=${size#*x}
  python scripts/evaluate.py --student "$RUN/final/student_ema" --out "$EVAL/$size" \
      --height "$h" --width "$w" --steps "$EVAL_STEPS" --batch_size 8 --pickscore \
      2>&1 | tee "logs/eval_continue_$size.log"
done
echo "done: metrics in $EVAL/*/metrics.md"
