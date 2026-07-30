@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Black-Box Scanner - Installation et lancement
echo ============================================
echo Ce script ne fait rien sur internet a part cloner le
echo depot et telecharger les paquets Python. Le site tourne
echo uniquement sur ton ordinateur (http://localhost:8501).
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERREUR] Git n'est pas installe.
    echo Telecharge-le ici : https://git-scm.com/download/win
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe.
    echo Telecharge-le ici : https://www.python.org/downloads/
    echo IMPORTANT : coche "Add Python to PATH" pendant l'installation.
    pause
    exit /b 1
)

if not exist "Black-Box" (
    echo Clonage du depot GitHub...
    git clone https://github.com/Angeltomas007/Black-Box.git
    if errorlevel 1 (
        echo [ERREUR] Le clonage a echoue. Verifie ta connexion internet.
        pause
        exit /b 1
    )
)

cd Black-Box
git checkout claude/algo-trading-black-box-quvnex
git pull origin claude/algo-trading-black-box-quvnex

if not exist "venv" (
    echo Creation de l'environnement virtuel Python...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installation des dependances (peut prendre quelques minutes la premiere fois)...
pip install --quiet --upgrade pip
pip install --quiet -r requirements-web.txt

echo.
echo ============================================
echo Lancement du scanner...
echo Ton navigateur va s'ouvrir sur http://localhost:8501
echo Ferme cette fenetre (ou Ctrl+C) pour arreter le site.
echo ============================================
echo.
streamlit run webapp\app.py

pause
