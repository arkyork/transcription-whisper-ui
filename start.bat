@echo off
setlocal

:: 仮想環境の Python 実行ファイルパス
set PYTHON_EXE=C:/Users/kijuh/AppData/Local/Programs/Python/Python312/python.exe

:: Pythonの仮想環境ディレクトリ名（既に作成済みを想定）
set VENV_DIR=.venv


echo [1] 仮想環境を作成中...
%PYTHON_EXE% -m venv %VENV_DIR%

echo [2] 仮想環境をアクティブ化...
call %VENV_DIR%\Scripts\activate.bat

echo [3] pip をアップグレード...
python -m pip install --upgrade pip

echo [4] requirements.txt を使って依存関係をインストール...
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python -m pip install -r requirements.txt

echo [5] 実行

python main.py

endlocal
pause
