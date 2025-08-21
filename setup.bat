@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing pip tools...
python -m pip install --upgrade pip setuptools wheel

echo Installing PyTorch first (largest dependency)...
pip install torch --index-url https://download.pytorch.org/whl/cu118

echo Installing other core dependencies...
pip install gymnasium mujoco stable-baselines3

echo Installing remaining requirements...
pip install tensorboard numpy

echo Installation complete!
echo.
echo To get started:
echo 1. Activate the environment: venv\Scripts\activate.bat
echo 2. Train the model: python train_ppo.py
echo 3. Evaluate the model: python eval_policy.py
