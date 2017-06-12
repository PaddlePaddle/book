#!/bin/bash
command -v go >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo >&2 "Please install go https://golang.org/doc/install#install"
    exit 1
fi

export GOPATH=~/go; go get -u github.com/wangkuiyi/ipynb/markdown-to-ipynb

cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#convert md to ipynb
for file in */{README,README\.cn}.md ; do
    ~/go/bin/markdown-to-ipynb < $file > ${file%.*}".ipynb"
    if [ $? -ne 0 ]; then
        echo >&2 "markdown-to-ipynb $file error"
        exit 1
    fi
done

if [[ -z $TEST_EMBEDDED_PYTHON_SCRIPTS ]]; then
    exit 0
fi

#exec ipynb's py file
for file in */{README,README\.cn}.ipynb ; do
    pushd $PWD > /dev/null
    cd $(dirname $file) > /dev/null

    echo "begin test $file"
    jupyter nbconvert --to python $(basename $file) --stdout | python

    popd > /dev/null
    #break
done
