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
        pattern = re.compile(
            'AvgCost=([0-9]+\.[0-9]+).*?Test.*? cost=([0-9]+\.[0-9]+).*?pass-([0-9]+)',
            re.S)
        results = re.findall(pattern, text)
        train_cost, test_cost, pass_ = zip(*results)
        train_cost_float = map(float, train_cost)
        test_cost_float = map(float, test_cost)
        pass_int = map(int, pass_)
        plt.plot(pass_int, train_cost_float, 'red', label='Train')
        plt.plot(pass_int, test_cost_float, 'g--', label='Test')
        plt.ylabel('AvgCost')
        plt.xlabel('Epoch')

        # Now add the legend with some customizations.
        legend = plt.legend(loc='upper right', shadow=False)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

        plt.show()


if __name__ == '__main__':
    plot_log(sys.argv[1])
