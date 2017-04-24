#!/bin/bash
set -xe

pip install notedown

cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#convert md to ipynb
for file in */{README,README\.en}.md ; do
    notedown $file > ${file%.*}.ipynb
done
