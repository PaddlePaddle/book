#!/bin/bash

function find_line() {
  local fn=$1
  local x=0
  cat $fn | while read line; do
    local x=$(( x+1 ))
    if echo $line | grep '${MARKDOWN}' -q; then
      echo $x
      break
    fi
  done
}

MD_FILE=$1
TMPL_FILE=$2
TPL_LINE=`find_line $TMPL_FILE`
cat $TMPL_FILE | head -n $((TPL_LINE-1))
cat $MD_FILE
cat $TMPL_FILE | tail -n +$((TPL_LINE+1))
