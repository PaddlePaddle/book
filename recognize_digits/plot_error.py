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

import matplotlib.pyplot as plt
import re
import sys


def plot_log(filename):
    with open(filename, 'r') as f:
        text = f.read()
        pattern = re.compile('Test.*? cost=([0-9]+\.[0-9]+).*?pass-([0-9]+)',
                             re.S)
        results = re.findall(pattern, text)
        cost, pass_ = zip(*results)
        cost_float = map(float, cost)
        pass_int = map(int, pass_)
        plt.plot(pass_int, cost_float, 'bo', pass_, cost_float, 'k')
        plt.ylabel('AvgCost')
        plt.xlabel('epoch')
        plt.show()


if __name__ == '__main__':
    plot_log(sys.argv[1])
