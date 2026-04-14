@echo off
chcp 65001 >nul
title СІМІ Dashboard

:: ── Перевірка Python ─────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ╔══════════════════════════════════════════════════════╗
    echo  ║   Python не знайдено!                                ║
    echo  ║                                                      ║
    echo  ║   1. Відкрийте https://www.python.org/downloads/    ║
    echo  ║   2. Завантажте Python 3.11 або новіший             ║
    echo  ║   3. При встановленні ОБОВ'ЯЗКОВО позначте:         ║
    echo  ║      [v] Add Python to PATH                         ║
    echo  ║   4. Перезапустіть цей файл                         ║
    echo  ╚══════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

:: ── Встановлення/оновлення пакетів ───────────────────────────
echo  Перевірка залежностей...
pip install --quiet --upgrade streamlit plotly pandas openpyxl numpy matplotlib kaleido 2>nul
if errorlevel 1 (
    echo  Встановлення пакетів...
    pip install streamlit plotly pandas openpyxl numpy matplotlib kaleido
)

:: ── Запуск ───────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   🟣 СІМІ Dashboard запускається...                  ║
echo  ║   Браузер відкриється автоматично.                   ║
echo  ║   Щоб зупинити — закрийте це вікно.                  ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

python launcher.py

pause
