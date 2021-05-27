#!/bin/bash
python3 -m venv ./venv
source ./venv/bin/activate
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine
#For generating images for readme
python3 -m pip install matplotlib
python3 -m pip install numpy
python3 -m pip install scipy
