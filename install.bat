@echo off
echo === ComfyUI JSON Editor Installer ===

set PYTHON_PATH=C:\Users\Startklar\AppData\Local\Programs\Python\Python312\python.exe

echo Erstelle virtuelles Environment (venv)...
%PYTHON_PATH% -m venv jsonvenv

echo Aktiviere venv und installiere Requirements...
call jsonvenv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo Installation abgeschlossen!
pause
