#!/bin/bash
command -v go >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo >&2 "Please install go https://golang.org/doc/install#install"
    exit 1
fi

export GOPATH=~/go; go get -u github.com/wangkuiyi/ipynb/markdown-to-ipynb

cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#convert md to ipynb
for file in */{README,README\.en}.md ; do
    ~/go/bin/markdown-to-ipynb < $file > ${file%.*}".ipynb"
    if [[ $? -ne 0 ]]; then
        echo >&2 "markdown-to-ipynb $file error"
        exit 1
    fi
done

if [[ -z $TEST_EMBEDDED_PYTHON_SCRIPTS ]]; then
    exit 0
fi

#exec ipynb's py file
for file in */{README,README\.en}.ipynb ; do
    pushd $PWD > /dev/null
    cd $(dirname $file) > /dev/null

    echo "begin test $file"
    if [[ $(dirname $file) == "08.recommender_system" ]]; then
       timeout -s SIGKILL 30 bash -c \
           "jupyter nbconvert --to python $(basename $file) --stdout  | \
           sed  's/get_ipython()\.magic(.*'\''matplotlib inline'\'')/\#matplotlib inline/g' | \
           sed '/^# coding: utf-8/a\import matplotlib\nmatplotlib.use('\''Agg'\'')' | python"
    else
       timeout -s SIGKILL 30  bash -c "jupyter nbconvert --to python $(basename $file) --stdout | python" 
    fi

    if [[ $? -ne 0 && $? -ne 124 && $? -ne 137 ]]; then
        echo >&2 "exec $file error!"
        exit 1
    fi

    popd > /dev/null
    #break
done
