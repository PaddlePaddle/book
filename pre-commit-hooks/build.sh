#!/bin/bash

for i in $(du -a | grep '\.\/.\+\/README.md' | cut -f 2); do
    .tmpl/convert-markdown-into-html.sh $i > $(dirname $i)/index.html
done

for i in $(du -a | grep '\.\/.\+\/README.en.md' | cut -f 2); do
    .tmpl/convert-markdown-into-html.sh $i > $(dirname $i)/index.en.html
done
