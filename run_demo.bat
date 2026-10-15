@echo off
echo ========================================================
echo  Launching Neuromorphic AI Live ECG Demo (Streamlit)
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
