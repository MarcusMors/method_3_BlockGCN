#!/bin/bash
set -e

echo "=========================================="
echo "BlockGCN Full Confusion Matrix Analysis"
echo "=========================================="

# ---- X-SUB ----
echo ""
echo ">>> X-Sub | Individual Modalities"
mkdir -p analysis/xsub

for mod in joint bone vel bone_vel; do
    echo "    Analyzing $mod..."
    # Map modality to directory and epoch
    case $mod in
        joint)      DIR=work_dir/ntu60/xsub/joint;      EPOCH=139 ;;
        bone)       DIR=work_dir/ntu60/xsub/bone;       EPOCH=138 ;;
        vel)        DIR=work_dir/ntu60/xsub/vel;        EPOCH=137 ;;
        bone_vel)   DIR=work_dir/ntu60/xsub/bone_vel;   EPOCH=117 ;;
    esac
    python analyze_modality_confusion.py \
        --pkl $DIR/epoch${EPOCH}_test_score.pkl \
        --split xsub --modality $mod \
        --output analysis/xsub/$mod
done

echo ">>> X-Sub | Ensemble"
python analyze_ensemble_confusion.py \
    --split xsub \
    --joint-dir work_dir/ntu60/xsub/joint \
    --bone-dir work_dir/ntu60/xsub/bone \
    --joint-motion-dir work_dir/ntu60/xsub/vel \
    --bone-motion-dir work_dir/ntu60/xsub/bone_vel \
    --joint-epoch 139 --bone-epoch 138 \
    --joint-motion-epoch 137 --bone-motion-epoch 117 \
    --alpha 0.6 0.7 0.35 0.2 \
    --output analysis/xsub/ensemble

# ---- X-VIEW ----
echo ""
echo ">>> X-View | Individual Modalities"
mkdir -p analysis/xview

for mod in joint bone vel bone_vel; do
    echo "    Analyzing $mod..."
    case $mod in
        joint)      DIR=work_dir/ntu60/xview/joint;      EPOCH=140 ;;
        bone)       DIR=work_dir/ntu60/xview/bone;       EPOCH=131 ;;
        vel)        DIR=work_dir/ntu60/xview/vel;        EPOCH=137 ;;
        bone_vel)   DIR=work_dir/ntu60/xview/bone_vel;   EPOCH=125 ;;
    esac
    python analyze_modality_confusion.py \
        --pkl $DIR/epoch${EPOCH}_test_score.pkl \
        --split xview --modality $mod \
        --output analysis/xview/$mod
done

echo ">>> X-View | Ensemble"
python analyze_ensemble_confusion.py \
    --split xview \
    --joint-dir work_dir/ntu60/xview/joint \
    --bone-dir work_dir/ntu60/xview/bone \
    --joint-motion-dir work_dir/ntu60/xview/vel \
    --bone-motion-dir work_dir/ntu60/xview/bone_vel \
    --joint-epoch 140 --bone-epoch 131 \
    --joint-motion-epoch 137 --bone-motion-epoch 125 \
    --alpha 0.6 0.7 0.35 0.2 \
    --output analysis/xview/ensemble

echo ""
echo "=========================================="
echo "Analysis complete! Check the analysis/ folder."
echo "=========================================="
