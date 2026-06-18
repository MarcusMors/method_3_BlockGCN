##!/bin/bash

# from 4.4 Training in  https://www.kimi.com/chat/19e4a82f-2c32-8638-8000-093f0444c9cd?chat_enter_method=home
python main.py --config config/nturgbd-cross-subject/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/xsub/joint --device 0 --batch-size 64 --num-worker 4

# original
# python main.py --config config/nturgbd-cross-subject/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic --device 0 --batch-size 64


##
#python main.py --config config/nturgbd-cross-subject/vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic --device 0
##
#python main.py --config config/nturgbd-cross-subject/bone.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic --device 0
##
#python main.py --config config/nturgbd-cross-subject/bone_vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic --device 0



# deterministic=true
#python main.py --config config/nturgbd-cross-view/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic --device 0
#
#python main.py --config config/nturgbd-cross-view/vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic --device 0
#
#python main.py --config config/nturgbd-cross-view/bone.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic --device 0
#
#python main.py --config config/nturgbd-cross-view/bone_vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic --device 0


# Phase 1 — NTU120 Cross-Subject
# this one, ensemble 1
# python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic --device 0



# this one, ensemble 2
#python main.py --config config/nturgbd120-cross-subject/vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic --device 0

# this one, ensemble 3
#python main.py --config config/nturgbd120-cross-subject/bone.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic --device 0

# this one, ensemble 4
#python main.py --config config/nturgbd120-cross-subject/bone_vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic --device 0
# then ensemble
# python ensemble.py --dataset ntu120 --split xsub


#python main.py --config config/nturgbd120-cross-set/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic --device 0
#
#python main.py --config config/nturgbd120-cross-set/vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic --device 0
#
#python main.py --config config/nturgbd120-cross-set/bone.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic --device 0

#python main.py --config config/nturgbd120-cross-set/bone_vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic --device 0


# python main.py --config config/ucla/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic --device 0
#
#python main.py --config config/ucla/vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic --device 0
#
#python main.py --config config/ucla/bone.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic --device 0

#python main.py --config config/ucla/bone_vel.yaml --model model.BlockGCN.Model --work-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic --device 0


