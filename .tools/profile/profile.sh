#!/bin/bash

book_dir=/book
work_dir=/book/.tools/profile
pushd $book_dir
array=()
while IFS=  read -r -d $'\0'; do
    array+=("$REPLY")
done < <(find ${book_dir} -name train.py -print0)

for file in "${array[@]}"; do
    echo $file
    dir=$(dirname "${file}")
    result=${file%.*}'.prof'
    result_png=${result}'.png'
    pushd $dir
    python -m cProfile -o $result $file
    popd
    ${work_dir}/get_stats.py $result
    gprof2dot -f pstats $result | dot -Tpng -o $result_png
done
popd
