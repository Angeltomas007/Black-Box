#!/bin/bash
# Double-click this file in Finder to install and launch the Black-Box
# Scanner locally. Nothing runs anywhere but this machine.
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Black-Box Scanner - Installation et lancement"
echo "============================================"
echo "Rien n'est mis en ligne : le depot est clone et les paquets"
echo "Python installes, mais le site tourne uniquement sur cet"
echo "ordinateur (http://localhost:8501)."
echo

if ! command -v git &> /dev/null; then
    echo "[ERREUR] Git n'est pas installe."
    echo "Lance 'xcode-select --install' dans le Terminal, puis relance ce script."
    read -p "Appuie sur Entree pour fermer..."
    exit 1
fi

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &> /dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERREUR] Python 3 n'est pas installe."
    echo "Telecharge-le ici : https://www.python.org/downloads/"
    read -p "Appuie sur Entree pour fermer..."
    exit 1
fi

if [ ! -d "Black-Box" ]; then
    echo "Clonage du depot GitHub..."
    git clone https://github.com/Angeltomas007/Black-Box.git
fi

cd Black-Box
git checkout claude/algo-trading-black-box-quvnex
git pull origin claude/algo-trading-black-box-quvnex

if [ ! -d "venv" ]; then
    echo "Creation de l'environnement virtuel Python..."
    "$PYTHON_BIN" -m venv venv
fi
source venv/bin/activate

echo "Installation des dependances (peut prendre quelques minutes la premiere fois)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-web.txt

echo
echo "============================================"
echo "Lancement du scanner..."
echo "Ton navigateur va s'ouvrir sur http://localhost:8501"
echo "Ferme cette fenetre (ou Ctrl+C) pour arreter le site."
echo "============================================"
echo
streamlit run webapp/app.py

read -p "Appuie sur Entree pour fermer..."
