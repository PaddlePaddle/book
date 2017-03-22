#!/bin/bash

set -e
# Conver markdown to ipynb
/book/.tools/convert-markdown-into-ipynb-and-test.sh

# Cache dataset
/book/.tools/cache_dataset.py
