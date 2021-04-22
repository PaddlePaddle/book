#!/bin/bash
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
set -e

DIR="$(cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

#Download MQ2007 dataset
echo "Downloading query-docs data..."
wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar

#Extract package
echo "Unzipping..."
unrar x MQ2007.rar

#Remove compressed package
rm MQ2007.rar

echo "data/MQ2007/Fold1/train.txt" > train.list
echo "data/MQ2007/Fold1/vali.txt" > test.list
echo "data/MQ2007/Fold1/test.txt" > pred.list

echo "Done."
