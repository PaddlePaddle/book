# MNIST classification by PaddlePaddle

Forked from https://github.com/sugyan/tensorflow-mnist

![screencast](https://cloud.githubusercontent.com/assets/80381/11339453/f04f885e-923c-11e5-8845-33c16978c54d.gif)

## Build

    $ docker build -t paddle-mnist .

## Usage


1. Download `inference_topology.pkl` and `param.tar` to current directory
1. Run following commands:
```bash
docker run -v `pwd`:/data -d -p 8000:80 -e WITH_GPU=0 paddlepaddle/book:serve
docker run -it -p 5000:5000 paddlepaddle/book:mnist
```
1. Visit http://localhost:5000
