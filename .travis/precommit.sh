#!/bin/bash
function abort(){
    echo "Your commit not fit PaddlePaddle code style" 1>&2
    echo "Please use pre-commit scripts to auto-format your code" 1>&2
    exit 1
}

trap 'abort' 0
set -e
cd `dirname $0`
cd ..
export PATH=/usr/bin:$PATH
export BRANCH=develop
pre-commit install

for file_name in `git diff --numstat upstream/$BRANCH |awk '{print $NF}'`;do
        if ! pre-commit run --files $file_name ; then
            commit_files=off
        fi
    done 
    
if [ $commit_files == 'off' ];then
    ls -lh
    git diff 2>&1
    exit 1
fi

trap : 0
