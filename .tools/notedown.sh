#!/bin/bash
set -xe

cd /book

#convert md to ipynb
for file in */{README,README\.en}.md ; do
    notedown $file > ${file%.*}.ipynb
done
