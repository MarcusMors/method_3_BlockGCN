@echo off
set PYTHON=python
set CONFIG_ROOT=config\nturgbd-cross-subject
set WORK_ROOT=work_dir\ntu60\xsub

REM Joint modality
%PYTHON% main_windows.py --config %CONFIG_ROOT%\default.yaml --model model.BlockGCN.Model --work-dir %WORK_ROOT%\joint --device 0 --batch-size 64 --num-worker 4

REM Bone modality
%PYTHON% main_windows.py --config %CONFIG_ROOT%\bone.yaml --model model.BlockGCN.Model --work-dir %WORK_ROOT%\bone --device 0 --batch-size 64 --num-worker 4

REM Joint velocity
%PYTHON% main_windows.py --config %CONFIG_ROOT%\vel.yaml --model model.BlockGCN.Model --work-dir %WORK_ROOT%\vel --device 0 --batch-size 64 --num-worker 4

REM Bone velocity
%PYTHON% main_windows.py --config %CONFIG_ROOT%\bone_vel.yaml --model model.BlockGCN.Model --work-dir %WORK_ROOT%\bone_vel --device 0 --batch-size 64 --num-worker 4

echo All training completed!