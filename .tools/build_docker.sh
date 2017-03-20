#!/bin/bash
cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd $cur_path/../

#convert md to ipynb
.tools/convert-markdown-into-ipynb-and-test.sh

paddle_version=0.10.0rc2

#generate docker file
if [ ${USE_UBUNTU_REPO_MIRROR} ]; then
  UPDATE_MIRROR_CMD="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  UPDATE_MIRROR_CMD="\\"
fi

mkdir -p build
cat > build/Dockerfile <<EOF
FROM paddlepaddle/paddle:${paddle_version}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN ${UPDATE_MIRROR_CMD}
        apt-get install locales
RUN localedef -f UTF-8 -i en_US en_US.UTF-8

RUN  apt-get -y install gcc && \
        apt-get -y clean

RUN pip install -U matplotlib jupyter numpy requests scipy

COPY . /book
RUN rm -rf /book/build

EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True /book/"]
EOF

#build docker image
echo "paddle_version:"$paddle_version
docker build --no-cache -t paddlepaddle/book:${paddle_version}  -t paddlepaddle/book:latest  -f ./build/Dockerfile .
