# Breakout

poetry is not working. gym does not install with atari option.

```bash
# venv setup
python3.10 -m venv nr-ml-venv
pip install -U pip setuptools wheel
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install stable-baselines3[extra]
```

```
# setup.py is from python packaging project sample
https://raw.githubusercontent.com/pypa/sampleproject/main/setup.py
```
