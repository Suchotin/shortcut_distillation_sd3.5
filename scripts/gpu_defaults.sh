# Per-GPU defaults, sourced by run_all.sh and setup_env.sh. Override with env vars.
#   >= 60 GB (80 GB-class GPUs): no gradient checkpointing (~25% faster), generation batch 16
#   <  60 GB (4090/5090/...):  gradient checkpointing on, generation batch 8
_GPU_MEM_MB=$({ nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true; } | head -n1 | tr -dc '0-9' || true)
_GPU_MEM_MB=${_GPU_MEM_MB:-0}
if [ "$_GPU_MEM_MB" -ge 60000 ]; then _GC=0; _GEN_BS=16; else _GC=1; _GEN_BS=8; fi
GC=${GC:-$_GC}
GEN_BS=${GEN_BS:-$_GEN_BS}
if [ "$GC" = 1 ]; then GC_FLAG=--gradient_checkpointing; else GC_FLAG=; fi
echo "[gpu_defaults] GPU memory ${_GPU_MEM_MB} MB -> gradient checkpointing=$GC, generation batch=$GEN_BS" >&2
