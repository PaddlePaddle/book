### This file is only used for continuous evaluation test!
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import AccKpi

conv_train_cost_kpi = CostKpi(
    'conv_train_cost', 0.02, 0, actived=True, desc='train cost')
conv_train_acc_kpi = AccKpi(
    'conv_train_acc', 0.02, 0, actived=True, desc='train acc')
conv_test_cost_kpi = CostKpi(
    'conv_test_cost', 0.02, 0, actived=True, desc='test cost')
conv_test_acc_kpi = AccKpi(
    'conv_test_acc', 0.02, 0, actived=True, desc='test acc')

rnn_train_cost_kpi = CostKpi(
    'rnn_train_cost', 0.02, 0, actived=True, desc='train cost')
rnn_train_acc_kpi = AccKpi(
    'rnn_train_acc', 0.02, 0, actived=True, desc='train acc')
rnn_test_cost_kpi = CostKpi(
    'rnn_test_cost', 0.02, 0, actived=True, desc='test cost')
rnn_test_acc_kpi = AccKpi(
    'rnn_test_acc', 0.02, 0, actived=True, desc='test acc')

lstm_train_cost_kpi = CostKpi(
    'lstm_train_cost', 0.02, 0, actived=True, desc='train cost')
lstm_train_acc_kpi = AccKpi(
    'lstm_train_acc', 0.02, 0, actived=True, desc='train acc')
lstm_test_cost_kpi = CostKpi(
    'lstm_test_cost', 0.02, 0, actived=True, desc='test cost')
lstm_test_acc_kpi = AccKpi(
    'lstm_test_acc', 0.02, 0, actived=True, desc='test acc')

tracking_kpis = [
    conv_train_cost_kpi, conv_train_acc_kpi, conv_test_cost_kpi,
    conv_test_acc_kpi, rnn_train_cost_kpi, rnn_train_acc_kpi, rnn_test_cost_kpi,
    rnn_test_acc_kpi, lstm_train_cost_kpi, lstm_train_acc_kpi,
    lstm_test_cost_kpi, lstm_test_acc_kpi
]


def parse_log(log):
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi
    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
