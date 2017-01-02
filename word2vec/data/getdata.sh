#!/bin/bash
set -e

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -zxf simple-examples.tgz
echo `pwd`/simple-examples/data/ptb.train.txt > train.list
echo `pwd`/simple-examples/data/ptb.valid.txt > test.list
