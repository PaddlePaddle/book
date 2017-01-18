#!/bin/bash

for file in `find . -name '*.md' | grep -v '^./README.md'`
do
	bash .tmpl/build.sh $file .tmpl/template.html > `dirname $file`/index.html
done
