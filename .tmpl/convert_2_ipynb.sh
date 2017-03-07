tmpl_path=$(cd "$(dirname "$0")"; pwd)
cd $tmpl_path/../

#define colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

#check go installed?
command -v go >/dev/null 2>&1 || { echo -e >&2 "${RED}I require go but it's not installed. Aborting.${NC}"; exit 1; }


#convert to ipynb file
export GOPATH=$PWD/.tmpl/go_tool/
go get -u github.com/wangkuiyi/ipynb/markdown-to-ipynb

for file in `ls */README*.md`
do
	name=${file%.*}".ipynb"
	$GOPATH/bin/markdown-to-ipynb < $file > $name
    if [ $? != 0 ]; then
        echo -e "${RED}markdown-to-ipynb $file error${NC}"
        exit 1
    fi
done


if [ -z $TEST_ALL_README ]; then
    exit 0
fi


#TODO check README.en.ipynb?
mkdir -p $tmpl_path/test_ipynb
echo "begin test all REAME.ipynb"
for file in `ls */README.ipynb`
do
    echo -e "${GREEN}begin test ${file}${NC}"
    dir_name=${file%/*}
    file_name=$(basename $file)
    dest_file_name=$(basename $file .ipynb).py

    dest_path=$tmpl_path/test_ipynb/$dir_name
    dest_file_path=$tmpl_path/test_ipynb/$dir_name/$dest_file_name

    mkdir -p $dest_path

    #convert to py file
    pushd $PWD
    cd $dir_name
    ipython nbconvert --to python $file_name --output=$dest_file_path

    #exec py file
    pushd $PWD
    python $dest_file_path
    if [ $? != 0 ]; then
        echo -e "${RED}run py file $file error${NC}"
        exit 1
    fi
    popd


    popd
    echo -e "${GREEN}complete test ${file}${NC}"
    #break
done
