#!/usr/bin/env sh
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This scripts downloads the mnist data and unzips it.
set -e
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
rm -rf "$DIR/raw_data"
mkdir "$DIR/raw_data"
cd "$DIR/raw_data"

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done

cd $DIR
rm -f *.list
echo "./data/raw_data/train" > "$DIR/train.list"
echo "./data/raw_data/t10k" > "$DIR/test.list"
