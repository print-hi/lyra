#!/bin/bash
rm -r -f lyra/__core 
rm -r build

mkdir lyra/__core
mkdir build/

python setup.py build
sudo find . -type f \( -iname '*.so' \) -exec cp {} lyra/__core/ \;
pytest