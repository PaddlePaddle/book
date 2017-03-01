#!/bin/bash

for file in `find . -name '*.md' | grep -v '^./README.md'`
do
	bash .tmpl/convert-markdown-into-html.sh $file > `dirname $file`/index.html
done
