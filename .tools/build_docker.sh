#!/bin/bash
cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#convert md to ipynb
.tools/convert-markdown-into-ipynb-and-test.sh

paddle_tag=0.10.0rc2
latest_tag=latest

#generate docker file
if [ ${USE_UBUNTU_REPO_MIRROR} ]; then
  update_mirror_cmd="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  update_mirror_cmd="\\"
fi

mkdir -p build

cat > ./build/Dockerfile << EOF
FROM paddlepaddle/paddle:${paddle_tag}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

COPY . /book
RUN python /book/.tools/cache_dataset.py

RUN ${update_mirror_cmd}
    apt-get update && \
    apt-get install -y locales && \
    apt-get -y install gcc && \
    apt-get -y clean && \
    localedef -f UTF-8 -i en_US en_US.UTF-8 && \
    pip install -U matplotlib jupyter numpy requests scipy

EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True /book/"]
EOF

#build docker image
echo "paddle_tag:"$paddle_tag
echo $dockerfile | docker build --no-cache -t paddlepaddle/book:${paddle_tag}  -t paddlepaddle/book:${latest_tag}  -f ./build/Dockerfile .
