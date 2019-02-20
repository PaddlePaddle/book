# Deep Learning with PaddlePaddle

[![Build Status](https://travis-ci.org/PaddlePaddle/book.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/book)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/book/blob/develop/README.md)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/book/blob/develop/README.cn.md)

1. [Fit a Line](http://www.paddlepaddle.org/documentation/book/en/develop/01.fit_a_line/index.html)
1. [Recognize Digits](http://www.paddlepaddle.org/documentation/book/en/develop/02.recognize_digits/index.html)
1. [Image Classification](http://www.paddlepaddle.org/documentation/book/en/develop/03.image_classification/index.html)
1. [Word to Vector](http://www.paddlepaddle.org/documentation/book/en/develop/04.word2vec/index.html)
1. [Recommender System](http://www.paddlepaddle.org/documentation/book/en/develop/05.recommender_system/index.html)
1. [Understand Sentiment](http://www.paddlepaddle.org/documentation/book/en/develop/06.understand_sentiment/index.html)
1. [Label Semantic Roles](http://www.paddlepaddle.org/documentation/book/en/develop/07.label_semantic_roles/index.html)
1. [Machine Translation](http://www.paddlepaddle.org/documentation/book/en/develop/08.machine_translation/index.html)

## Running the Book

This book you are reading is interactive -- each chapter can run as a Jupyter Notebook.

We packed this book, Jupyter, PaddlePaddle, and all dependencies into a Docker image. So you don't need to install anything except Docker. If you are using Windows, please follow [this installation guide](https://www.docker.com/docker-windows).  If you are running Mac, please follow [this](https://www.docker.com/docker-mac). For various Linux distros, please refer to https://www.docker.com.  If you are using Windows or Mac, you might want to give Docker [more memory and CPUs/cores](http://stackoverflow.com/a/39720010/724872).

Just type

```bash
docker run -d -p 8888:8888 paddlepaddle/book

```

This command will download the pre-built Docker image from DockerHub.com and run it in a container.  Please direct your Web browser to http://localhost:8888 to read the book.

If you are living in somewhere slow to access DockerHub.com, you might try our mirror server hub.baidubce.com:

```bash
docker run -d -p 8888:8888 hub.baidubce.com/paddlepaddle/book

```

### Training with GPU

By default we are using CPU for training, if you want to train with GPU, the steps are a little different.

To make sure GPU can be successfully used from inside container, please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Then run:

```bash
nvidia-docker run -d -p 8888:8888 paddlepaddle/book:latest-gpu

```

Or you can use the image registry mirror in China:

```bash
nvidia-docker run -d -p 8888:8888 hub.baidubce.com/paddlepaddle/book:latest-gpu

```

Change the code in the chapter that you are reading from
```python
use_cuda = False
```

to:
```python
use_cuda = True
```


## Contribute

Your contribution is welcome!  Please feel free to file Pull Requests to add your chapter as a directory under `/pending`. Once it is going stable, the community would like to move it to `/`.

To write, run, and debug your chapters, you will need Python 2.x, Go >1.5. You can build the Docker image using [this script](https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh).
This tutorial is contributed by <a xmlns:cc="http://creativecommons.org/ns#" href="http://www.paddlepaddle.org/" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a>, and licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
