#!/bin/bash
cur_path=$(dirname $(readlink -f $0))
cd $cur_path/../

#convert md to ipynb
./tools/convert-markdown-into-ipynb-and-test.sh

#get submodule
git submodule update --init --recursive
cd paddle && git checkout develop && paddle_version=`git describe --abbrev=0 --tags` && cd ..
if [ $? -ne 0 ]; then
	echo 1>&2 "get paddle version error"
	exit 1
fi

#generate docker file
if [ ${USE_MIRROR} ]; then
  MIRROR_UPDATE="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  MIRROR_UPDATE="\\"
fi

mkdir -p build
cat > build/Dockerfile <<EOF
FROM paddledev/paddle:${paddle_version}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN apt-get install locales 
RUN localedef -f UTF-8 -i en_US en_US.UTF-8

RUN ${MIRROR_UPDATE}
	apt-get -y install gcc && \
	apt-get -y clean

RUN pip install -U matplotlib jupyter numpy requests

COPY . /book
RUN rm -rf /book/build

EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 /book/"]
EOF

#build docker image
echo "paddle_version:"$paddle_version
docker build -t paddledev/book:${paddle_version}  -t paddledev/book:latest  -f ./build/Dockerfile .
