@echo off
REM Train all models WITH dropout sequentially
REM Usage: scripts\train_all_dropout.bat

echo ==========================================
echo Starting Ablation Study - With Dropout
echo ==========================================
echo.

echo [%date% %time%] Training M1 Small Baseline (Dropout)...
python scripts/train_model.py --config config/model_configs/m1_small_baseline_dropout.yaml
echo.

echo [%date% %time%] Training M2 Small Simple Attention (Dropout)...
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn_dropout.yaml
echo.

echo [%date% %time%] Training M3 Medium Baseline (Dropout)...
python scripts/train_model.py --config config/model_configs/m3_medium_baseline_dropout.yaml
echo.

echo [%date% %time%] Training M4 Medium Simple Attention (Dropout)...
python scripts/train_model.py --config config/model_configs/m4_medium_simple_attn_dropout.yaml
echo.

echo [%date% %time%] Training M5 Medium Additive Attention (Dropout)...
python scripts/train_model.py --config config/model_configs/m5_medium_additive_attn_dropout.yaml
echo.

echo [%date% %time%] Training M6 Medium Scaled DP Attention (Dropout)...
python scripts/train_model.py --config config/model_configs/m6_medium_scaled_dp_attn_dropout.yaml
echo.

echo ==========================================
echo All training completed!
echo [%date% %time%]
echo ==========================================
pause
