#!/bin/bash
cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#check cache data
cache_data_path=.cache/paddle/dataset
if [ ${COPY_CACHE_DATA} ] && [ ! -d $cache_data_path ];  then
  echo 2>&1 "Check the cache_data_path:${cache_data_path}"
  exit 1
fi

#convert md to ipynb
.tools/convert-markdown-into-ipynb-and-test.sh

paddle_version=0.10.0rc2
latest_label=latest

#generate docker file
if [ ${USE_UBUNTU_REPO_MIRROR} ]; then
  update_mirror_cmd="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  update_mirror_cmd="\\"
fi

mkdir -p build
cat > build/Dockerfile <<EOF1
FROM paddlepaddle/paddle:${paddle_version}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN ${update_mirror_cmd}
        apt-get install locales
RUN localedef -f UTF-8 -i en_US en_US.UTF-8

RUN  apt-get -y install gcc && \
        apt-get -y clean

RUN pip install -U matplotlib jupyter numpy requests scipy

COPY . /book
RUN rm -rf /book/build
EOF1

if [ ${COPY_CACHE_DATA} ]; then

cat >> build/Dockerfile << EOF2
RUN mkdir -p /root/${cache_data_path}
RUN mv /book/${cache_data_path}/* /root/${cache_data_path}/ && rm -rf /book/${cache_data_path}
EOF2

fi

cat >> build/Dockerfile << EOF3

EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True /book/"]
EOF3

#build docker image
echo "paddle_version:"$paddle_version
docker build --no-cache -t paddlepaddle/book:${paddle_version}  -t paddlepaddle/book:${latest_label}  -f ./build/Dockerfile .
