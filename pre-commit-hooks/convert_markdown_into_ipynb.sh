#!/bin/sh
for file in $@ ; do
    markdown-to-ipynb < $file > ${file%.*}".ipynb"
    if [ $? -ne 0 ]; then
        echo >&2 "markdown-to-ipynb $file error"
        exit 1
    fi
done

