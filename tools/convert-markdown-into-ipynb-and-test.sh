#!/bin/sh
command -v go >/dev/null 2>&1
if [ $? != 0 ]; then
    echo >&2 "Please install go https://golang.org/doc/install#install"
    exit 1
fi

GOPATH=/tmp/go go get -u github.com/wangkuiyi/ipynb/markdown-to-ipynb

cur_path=$(dirname $(readlink -f $0))
cd $cur_path/../

#convert md to ipynb
for file in */{README,README\.en}.md ; do
    /tmp/go/bin/markdown-to-ipynb < $file > ${file%.*}".ipynb"
    if [ $? != 0 ]; then
        echo >&2 "markdown-to-ipynb $file error"
        exit 1
    fi
done

if [[ ! -z $TEST_EMBEDDED_PYTHON_SCRIPTS ]]; then
    exit 0
fi

#exec ipynb's py file
for file in */{README,README\.en}.ipynb ; do
    pushd $PWD > /dev/null
    cd $(dirname $file) > /dev/null

    echo "begin test $file"
    jupyter nbconvert --to python $(basename $file) --stdout | python

    popd > /dev/null
    #break
done
