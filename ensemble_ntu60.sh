#!/bin/bash
# =============================================================================
# ensemble_ntu60.sh
# =============================================================================
# Run the 4-stream BlockGCN ensemble for NTU60.
#
#   bash ensemble_ntu60.sh
#
# The script auto-finds the best epoch (<= 140) from each modality's log.txt,
# then fuses the four streams with the official alpha weights.
#
# If you want to override with specific epochs, edit the EPOCH_* lines below.
# =============================================================================

set -e

PROJECT_ROOT="/home/marcus/+projects/pfc/method_3_BlockGCN"
cd "$PROJECT_ROOT"

echo "========================================"
echo "BlockGCN NTU60 Ensemble"
echo "========================================"

# ---------------------------------------------------------------------------
# X-Sub (Cross-Subject)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running X-Sub ensemble..."

python ensemble_ntu60.py \
  --split xsub \
  --joint-dir       work_dir/ntu60/xsub/joint \
  --bone-dir        work_dir/ntu60/xsub/bone \
  --joint-motion-dir work_dir/ntu60/xsub/vel \
  --bone-motion-dir work_dir/ntu60/xsub/bone_vel \
  --max-epoch 140

# ---------------------------------------------------------------------------
# X-View (Cross-View) — uncomment when training is complete
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running X-View ensemble..."

python ensemble_ntu60.py \
  --split xview \
  --joint-dir       work_dir/ntu60/xview/joint \
  --bone-dir        work_dir/ntu60/xview/bone \
  --joint-motion-dir work_dir/ntu60/xview/vel \
  --bone-motion-dir work_dir/ntu60/xview/bone_vel \
  --max-epoch 140

echo ""
echo "========================================"
echo "Ensemble complete!"
echo "========================================"
