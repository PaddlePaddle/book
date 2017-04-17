#!/usr/local/bin/python

import pstats
import sys

file_readable = sys.argv[1] + ".readable"
stream = open(file_readable, "w")

p = pstats.Stats(sys.argv[1], stream=stream)
p.strip_dirs().sort_stats("time").print_stats(20)
p.strip_dirs().sort_stats("cumtime").print_stats(20)
stream.close()
