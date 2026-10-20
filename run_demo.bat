@echo off
title 4-Class SNN FPGA Live Demo
cd /d "%~dp0"
echo ========================================================
echo  Launching Neuromorphic SNN ECG FPGA Demonstration GUI
echo  Target: Xilinx Spartan-7 (XC7S15)
echo ========================================================
echo.
python fpga_demo\ecg_fpga_gui.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
