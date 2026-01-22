@echo off
REM Train all models without dropout sequentially
REM Usage: scripts\train_all_no_dropout.bat

echo ==========================================
echo Starting Ablation Study - No Dropout
echo ==========================================
echo.

echo [%date% %time%] Training M1 Small Baseline...
python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml
echo.

echo [%date% %time%] Training M2 Small Simple Attention...
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml
echo.

echo [%date% %time%] Training M3 Medium Baseline...
python scripts/train_model.py --config config/model_configs/m3_medium_baseline.yaml
echo.

echo [%date% %time%] Training M4 Medium Simple Attention...
python scripts/train_model.py --config config/model_configs/m4_medium_simple_attn.yaml
echo.

echo [%date% %time%] Training M5 Medium Additive Attention...
python scripts/train_model.py --config config/model_configs/m5_medium_additive_attn.yaml
echo.

echo [%date% %time%] Training M6 Medium Scaled DP Attention...
python scripts/train_model.py --config config/model_configs/m6_medium_scaled_dp_attn.yaml
echo.

echo ==========================================
echo All training completed!
echo [%date% %time%]
echo ==========================================
pause
