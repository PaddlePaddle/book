#!/bin/bash
set -xe

cur_path="$(cd "$(dirname "$0")" && pwd -P)"
cd "$cur_path"/../

#paddle production image name
if [ ! -n "$1" ]; then
  paddle_image=paddlepaddle/paddle
else
  paddle_image=$1
fi

#paddle production image tag
if [ ! -n "$2" ]; then
  paddle_tag=0.10.0
else
  paddle_tag=$2
fi

#paddle book image name
if [ ! -n "$3" ]; then
  book_image=paddlepaddle/book
else
  book_image=$3
fi

#paddle book image tag
if [ ! -n "$4" ]; then
  book_tag=latest
else
  book_tag=$4
fi

#generate docker file
if [ ${USE_UBUNTU_REPO_MIRROR} ]; then
  update_mirror_cmd="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  update_mirror_cmd="\\"
fi

#build docker image
echo "paddle_tag:"$paddle_tag
echo "book_tag:"$book_tag

cat > Dockerfile <<EOF
FROM ${paddle_image}:${paddle_tag}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

COPY . /book
EOF

if [ -n "${http_proxy}" ]; then
cat >> Dockerfile <<EOF
ENV http_proxy ${http_proxy}
ENV https_proxy ${http_proxy}
EOF
fi

cat >> Dockerfile <<EOF
RUN pip install -U nltk \
    && python /book/.tools/cache_dataset.py

RUN ${update_mirror_cmd}
    apt-get update && \
    apt-get install -y locales patch && \
    apt-get -y install gcc curl git vim && \
    apt-get -y clean && \
    localedef -f UTF-8 -i en_US en_US.UTF-8 && \
    pip install --upgrade pip && \
    pip install -U notedown pillow matplotlib jupyter numpy requests scipy

RUN curl https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz -o go1.8.linux-amd64.tar.gz && \
    tar -zxvf go1.8.linux-amd64.tar.gz -C /usr/local/ && \
    rm go1.8.linux-amd64.tar.gz

ENV GOROOT /usr/local/go
ENV PATH \${GOROOT}/bin:\${PATH}

#convert md to ipynb
RUN /bin/bash /book/.tools/convert-markdown-into-ipynb-and-test.sh

EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True /book/"]
EOF

docker build --no-cache -t ${book_image}:${paddle_tag} -t ${book_image}:${book_tag} .
