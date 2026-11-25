# Sourced by run_all.sh / run_continue.sh: GPUs with >= 60 GB train without gradient
# checkpointing and generate with batch 16; smaller GPUs use checkpointing and batch 8.
_MEM=$({ nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true; } | head -n1 | tr -dc '0-9')
if [ "${_MEM:-0}" -ge 60000 ]; then _GC=0; _GEN_BS=16; else _GC=1; _GEN_BS=8; fi
GC=${GC:-$_GC}
GEN_BS=${GEN_BS:-$_GEN_BS}
if [ "$GC" = 1 ]; then GC_FLAG=--gradient_checkpointing; else GC_FLAG=; fi
