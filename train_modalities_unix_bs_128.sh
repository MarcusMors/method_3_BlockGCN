#!/bin/bash
set -e

LR=0.0759
BS=128
EPOCHS=140
DEVICE=0

echo "========================================"
echo "BlockGCN Training - All Modalities"
echo "LR=$LR | BS=$BS | Epochs=$EPOCHS"
echo "========================================"

# ============================================================
# NTU60 CROSS-SUBJECT (xsub)
# ============================================================
echo ""
echo ">>> NTU60 Cross-Subject"

python main_unix.py \
  --config config/nturgbd-cross-subject/default.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xsub/joint \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd-cross-subject/vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xsub/vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd-cross-subject/bone.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xsub/bone \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
 --config config/nturgbd-cross-subject/bone_vel.yaml \
 --model model.BlockGCN.Model \
 --work-dir work_dir/ntu60/xsub/bone_vel \
 --device $DEVICE \
 --batch-size $BS \
 --test-batch-size $BS \
 --num-epoch $EPOCHS \
 --base-lr $LR

# ============================================================
# NTU60 CROSS-VIEW (xview)
# ============================================================
echo ""
echo ">>> NTU60 Cross-View"

python main_unix.py \
  --config config/nturgbd-cross-view/default.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xview/joint \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd-cross-view/vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xview/vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd-cross-view/bone.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xview/bone \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd-cross-view/bone_vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu60/xview/bone_vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

# ============================================================
# NTU120 CROSS-SUBJECT (xsub)
# ============================================================
echo ""
echo ">>> NTU120 Cross-Subject"

python main_unix.py \
  --config config/nturgbd120-cross-subject/default.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xsub/joint \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-subject/vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xsub/vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-subject/bone.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xsub/bone \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-subject/bone_vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xsub/bone_vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

# ============================================================
# NTU120 CROSS-SET (xset)
# ============================================================
echo ""
echo ">>> NTU120 Cross-Set"

python main_unix.py \
  --config config/nturgbd120-cross-set/default.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xset/joint \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-set/vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xset/vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-set/bone.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xset/bone \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

python main_unix.py \
  --config config/nturgbd120-cross-set/bone_vel.yaml \
  --model model.BlockGCN.Model \
  --work-dir work_dir/ntu120/xset/bone_vel \
  --device $DEVICE \
  --batch-size $BS \
  --test-batch-size $BS \
  --num-epoch $EPOCHS \
  --base-lr $LR

echo ""
echo "========================================"
echo "All 16 training runs complete!"
echo "========================================"
