#!/bin/bash
#python -m pip install --upgrade pip
#pip install black==24.8.0 flake8==7.1.1 pylint==3.3.1 mypy==1.11.2 pytest==8.3.3 pytest-asyncio==0.24.0 pytest-mock==3.14.0 
python -m black .
python -m black --check .
python -m flake8 .
python -m pylint *.py
python -m mypy .
python -m pytest tests/ -v --asyncio-mode=auto
