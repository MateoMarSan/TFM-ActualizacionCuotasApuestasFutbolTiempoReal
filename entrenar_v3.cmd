@echo off
REM Modelo v3 (convocatoria completa): entrenamiento, calibracion, evaluacion y demo.
cd /d "%~dp0"
python -u train.py > train_v3.log 2> train_v3_errores.log || exit /b 1
python -u train.py --calibrar >> train_v3.log 2>> train_v3_errores.log || exit /b 1
python -u evaluate.py > evaluate_v3.log 2>> train_v3_errores.log || exit /b 1
python -u train.py --demo > demo_v3.log 2>> train_v3_errores.log
python -u figuras_memoria.py >> demo_v3.log 2>> train_v3_errores.log
echo FIN > fin_v3.txt
