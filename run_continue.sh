#!/usr/bin/env bash
# Continue a run from its final checkpoint (weights, EMA, optimizer and step counter), optionally
# on extra teacher data, then evaluate.
#
#   BASE_RUN=$DATA_ROOT/runs/shortcut_1024   RUN_NAME=shortcut_1024_12k   STEPS=12000
#   DATA=$DATA_ROOT/data/sd35m_1024          the base run's data
#   EXTRA_PROMPTS=data/prompts_v3.txt        optional: generate and add more data (seed 1)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=${ROOT:-$HOME/shortcut_work}
# shellcheck disable=SC1091
[ -f "$ROOT/shortcut_env.sh" ] && source "$ROOT/shortcut_env.sh"
# shellcheck disable=SC1091
source scripts/gpu_defaults.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

DATA_ROOT=${DATA_ROOT:-$ROOT}
BASE_RUN=${BASE_RUN:?set BASE_RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
STEPS=${STEPS:?set STEPS (total, including the base run)}
DATA=${DATA:-$DATA_ROOT/data/sd35m_1024}
EXTRA_PROMPTS=${EXTRA_PROMPTS:-}
EXTRA_DATA=${EXTRA_DATA:-${DATA}_extra}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-2}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_SIZES=${EVAL_SIZES:-"1024x1024 832x480"}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
RUN=$DATA_ROOT/runs/$RUN_NAME
EVAL=$DATA_ROOT/eval/$RUN_NAME
[ -f "$BASE_RUN/final/train_state.pt" ] || { echo "no $BASE_RUN/final/train_state.pt"; exit 1; }
mkdir -p logs

if [ -n "$EXTRA_PROMPTS" ]; then
  read -r RES_W RES_H < <(python -c "import json; m = json.load(open('$DATA/meta.json')); print(m['width'], m['height'])")
  pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_data.py --prompts "$EXTRA_PROMPTS" \
        --out "$EXTRA_DATA" --width "$RES_W" --height "$RES_H" --batch_size "$GEN_BS" --seed 1 \
        --num_workers "$NGPU" --worker_id "$i" > "logs/gen_extra_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  # one directory with both shard sets, as symlinks
  MIX=${DATA}+$(basename "$EXTRA_DATA")
  mkdir -p "$MIX" && find "$MIX" -maxdepth 1 -type l -delete
  n=0
  for f in "$DATA"/shard_*.safetensors "$EXTRA_DATA"/shard_*.safetensors; do
    ln -s "$f" "$MIX/$(printf 'shard_%05d' $n).safetensors"
    n=$((n + 1))
  done
  cp "$DATA"/{null_embeds.safetensors,showcase.safetensors,showcase.json} "$MIX/"
  python - "$DATA/meta.json" "$EXTRA_DATA/meta.json" "$MIX/meta.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("model", "height", "width", "steps", "cfg", "shift", "latent_shape"):
    assert a[k] == b[k], f"teacher setting differs: {k}"
a["n"] += b["n"]
json.dump(a, open(sys.argv[3], "w"), indent=2)
EOF
  DATA=$MIX
fi

# shellcheck disable=SC2086
accelerate launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/train.py \
    --data_dir "$DATA" --output_dir "$RUN" --resume "$BASE_RUN/final" --seed 1 \
    --batch_size "$BATCH" --grad_accum "$ACCUM" --max_steps "$STEPS" $GC_FLAG $TRAIN_ARGS \
    2>&1 | tee "logs/train_$RUN_NAME.log"

for size in $EVAL_SIZES; do
  python scripts/evaluate.py --student "$RUN/final/student_ema" --out "$EVAL/$size" \
      --width "${size%x*}" --height "${size#*x}" --batch_size 8 --pickscore \
      2>&1 | tee "logs/eval_${RUN_NAME}_$size.log"
done
echo "done: $EVAL/*/metrics.md"
