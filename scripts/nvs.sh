#! /bin/bash
#
# Usage Examples
#
# Train with PRoPE on RealEstate10K with 2 GPUs and plucker ray encoding:
# bash ./scripts/nvs.sh --pe PRoPE --ray_encoding plucker --gpus "0,1" --dataset RealEstate10K
#
# Train with DPPEdual on MVImgNet2 with 2 GPUs (scaled model):
# bash ./scripts/nvs.sh --pe DPPEdual --ray_encoding none --gpus "0,1" --dataset MVImgNet2 --num_layers 12 --dim_feedforward 3072 --max_steps 320000
#
# Test with zoom-in:
# bash ./scripts/nvs.sh --pe DPPEdual --ray_encoding none --dataset MVImgNet2 --resume <path/to/checkpoint.pt> --num_layers 12 --dim_feedforward 3072 --test-zoom-in '1.5 2 3 5'
#
# Test with more context views:
# bash ./scripts/nvs.sh --pe DPPEdual --ray_encoding none --dataset MVImgNet2 --resume <path/to/checkpoint.pt> --num_layers 12 --dim_feedforward 3072 --test-context-views '4 6 8 10 12'


# Default values
# ==============================
SEED=42
DATASET="MVImgNet2"
RAY_ENCODING="none"
GPUS="0,1"
BATCH_SIZE="32"  # global batch size should be 64
LR="0.0005"
MAX_STEPS="320000"
DECAY_STEPS="0"
NUM_LAYERS="12"
DIM_FEEDFORWARD=3072
RES=256
# ==============================
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --ray_encoding)
      RAY_ENCODING="$2"
      shift 2
      ;;
    --pe)
      PE_NAME="$2"
      shift 2
      ;;
    --qk_pe)
      QK_PE="$2"
      shift 2
      ;;
    --vo_pe)
      VO_PE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --decay_steps)
      DECAY_STEPS="$2"
      shift 2
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    --dim_feedforward)
      DIM_FEEDFORWARD="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --mv_train)
      MV_TRAIN="$2"
      shift 2
      ;;
    --only_model)
      ONLY_MODEL=true
      shift 1
      ;;
    --test-zoom-in)
      TEST_ZOOM_IN="$2"
      shift 2
      ;;
    --test-context-views)
      TEST_CONTEXT_VIEWS="$2"
      shift 2
      ;;
    --res)
      RES="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  --pe: PE method name (RoPE, CAPE, GTA, PRoPE, DPPEdual, DPPEtAdd)"
      echo "  --qk_pe: qk positional encoding type (alternative to --pe)"
      echo "  --vo_pe: vo positional encoding type (alternative to --pe)"
      echo "  --ray_encoding: plucker, camray, none, or raymap"
      echo "  --gpus: comma-separated GPU list (e.g., '0,1')"
      echo "  --seed: random seed for reproducibility"
      echo "  --resume: checkpoint path for resuming or evaluation"
      echo "  --test-zoom-in: space-separated zoom factors for testing (e.g., '3 5')"
      echo "  --test-context-views: space-separated context views for testing (e.g., '2 4 8 16')"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Resolve PE configuration
if [ -n "$PE_NAME" ] && ([ -n "$QK_PE" ] || [ -n "$VO_PE" ]); then
  echo "Error: Cannot specify both --pe and --qk_pe/--vo_pe"
  exit 1
fi

if [ -n "$PE_NAME" ]; then
  case $PE_NAME in
    RoPE)
      QK_PE="2d"
      VO_PE="none"
      ;;
    CAPE)
      QK_PE="Rt"
      VO_PE="none"
      ;;
    GTA)
      QK_PE="Rt2d"
      VO_PE="Rt2d"
      ;;
    PRoPE)
      QK_PE="p2d"
      VO_PE="p2d"
      ;;
    DPPEdual)
      QK_PE="pIT2d"
      VO_PE="pIT2d"
      ;;
    DPPEtAdd)
      QK_PE="p2d"
      VO_PE="KRtAdd2d"
      ;;
    *)
      echo "Error: Unknown PE name '$PE_NAME'. Valid options: RoPE, CAPE, GTA, PRoPE, DPPEdual, DPPEtAdd"
      exit 1
      ;;
  esac
elif [ -z "$QK_PE" ] || [ -z "$VO_PE" ]; then
  echo "Error: Must specify either --pe or both --qk_pe and --vo_pe"
  exit 1
fi

# Check required arguments
if [ -z "$RAY_ENCODING" ]; then
  echo "Error: --ray_encoding is required"
  exit 1
fi

if [ -z "$GPUS" ]; then
  echo "Error: --gpus is required"
  exit 1
fi

NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Build experiment name
if [ -n "$PE_NAME" ]; then
  PE_LABEL="${PE_NAME}"
else
  PE_LABEL="qk:${QK_PE}-vo:${VO_PE}"
fi
NAME="${SEED}_${DATASET}-ray:${RAY_ENCODING}-pe:${PE_LABEL}-depth:${NUM_LAYERS}-lr:${LR}-step:${MAX_STEPS}"

if [ -n "$RESUME" ]; then
  if [ "$ONLY_MODEL" = true ]; then
    NAME="resumed_${NAME}"
  fi
fi

if [ -n "$MV_TRAIN" ]; then
  NAME="MViewNum_${NAME}"
  echo "Multi view num training enabled with range: $MV_TRAIN"
fi

BASE_CMD=(
    "OMP_NUM_THREADS=1 uv run torchrun -m --nnodes=${SLURM_NNODES:-1} --nproc-per-node=$NGPUS --node_rank=${SLURM_NODEID:-0} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}"
    "nvs.trainval lvsm"
    "--seed ${SEED}"
    "--amp --amp_dtype bf16"
    "--dataset ${DATASET}"
    "--dataset_batch_scenes ${BATCH_SIZE}"
    "--dataset_supervise_views 1"
    "--dataset_patch_size ${RES}"
    "--model_config.encoder.num_layers ${NUM_LAYERS}"
    "--model_config.encoder.layer.d_model 768"
    "--model_config.encoder.layer.nhead 16"
    "--model_config.encoder.layer.dim_feedforward ${DIM_FEEDFORWARD}"
    "--model_config.encoder.layer.qk_norm"
    "--model_config.img_shape ${RES} ${RES} 3"
    "--model_config.cam_shape ${RES} ${RES} 6"
    "--max_steps ${MAX_STEPS}"
    "--test_every 4000"
    "--print_every 50"
    "--visual_every 50"
    "--warmup_steps 500"
    "--decay_steps ${DECAY_STEPS}"
    "--lr ${LR}"
    "--model_config.ray_encoding ${RAY_ENCODING}"
    "--model_config.qk_pe ${QK_PE}"
    "--model_config.vo_pe ${VO_PE}"
    "--output_dir results/nvs/${NAME}"
)

if [ -n "$MV_TRAIN" ]; then
  # Pass without quotes so "2 4" becomes two separate args for tyro parsing
  BASE_CMD+=("--mv_train" ${MV_TRAIN})
fi

echo "NAME: ${NAME}"
echo "RAY_ENCODING: ${RAY_ENCODING}"
echo "QK_PE: ${QK_PE}"
echo "VO_PE: ${VO_PE}"
echo "DECAY_STEPS: ${DECAY_STEPS}"

# Determine if this is a test run
IS_TEST=false
if [ -n "$TEST_ZOOM_IN" ] || [ -n "$TEST_CONTEXT_VIEWS" ]; then
  IS_TEST=true
fi

if [ "$IS_TEST" = true ]; then
    # =====================================================================
    # [Testing Mode]
    # =====================================================================
    CURRENT_BASE_CMD=("${BASE_CMD[@]}")
    if [ -n "$RESUME" ]; then
        CURRENT_BASE_CMD+=("--resume ${RESUME}")
        if [ "$ONLY_MODEL" = true ]; then
            CURRENT_BASE_CMD+=("--only_model")
        fi
    fi

    if [ -n "$TEST_ZOOM_IN" ]; then
        for zoom_factor in $TEST_ZOOM_IN; do
            echo "Starting testing with zoom factor ${zoom_factor}..."
            CMD=(
                "${CURRENT_BASE_CMD[@]}"
                "--test_only --auto_resume"
                "--test_zoom_factor ${zoom_factor}"
                "--test_subdir eval${RES}-zoom${zoom_factor}x"
            )
            CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
        done
    elif [ -n "$TEST_CONTEXT_VIEWS" ]; then
        for context_views in $TEST_CONTEXT_VIEWS; do
            echo "Starting testing with ${context_views} context views..."
            CMD=(
                "${CURRENT_BASE_CMD[@]}"
                "--test_only --auto_resume"
                "--model_config.ref_views ${context_views}"
                "--test_input_views ${context_views}"
                "--test_subdir eval${RES}-context${context_views}"
            )
            CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
        done
    fi
    exit 0

else
    # =====================================================================
    # [Training Mode]
    # =====================================================================
    CURRENT_BASE_CMD=("${BASE_CMD[@]}")
    if [ -n "$RESUME" ]; then
      CURRENT_BASE_CMD+=("--resume ${RESUME}")
      if [ "$ONLY_MODEL" = true ]; then
        CURRENT_BASE_CMD+=("--only_model")
      fi
    fi

    echo "Starting training process..."
    CMD=(
        "${CURRENT_BASE_CMD[@]}"
    )
    CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    exit 0
fi
