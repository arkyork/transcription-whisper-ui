@echo off
setlocal

:: ���z���� Python ���s�t�@�C���p�X
set PYTHON_EXE=C:/Users/kijuh/AppData/Local/Programs/Python/Python312/python.exe

:: Python�̉��z���f�B���N�g�����i���ɍ쐬�ς݂�z��j
set VENV_DIR=.venv


echo [1] ���z�����쐬��...
%PYTHON_EXE% -m venv %VENV_DIR%

echo [2] ���z�����A�N�e�B�u��...
call %VENV_DIR%\Scripts\activate.bat

echo [3] pip ���A�b�v�O���[�h...
python -m pip install --upgrade pip

echo [4] requirements.txt ���g���Ĉˑ��֌W���C���X�g�[��...
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python -m pip install -r requirements.txt

echo [5] ���s

python main.py

endlocal
pause
