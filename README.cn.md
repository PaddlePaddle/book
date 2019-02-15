# 深度学习入门

[![Build Status](https://travis-ci.org/PaddlePaddle/book.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/book)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/book/blob/develop/README.md)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/book/blob/develop/README.cn.md)

1. [线性回归](http://www.paddlepaddle.org/documentation/book/zh/develop/01.fit_a_line/index.cn.html)
1. [识别数字](http://www.paddlepaddle.org/documentation/book/zh/develop/02.recognize_digits/index.cn.html)
1. [图像分类](http://www.paddlepaddle.org/documentation/book/zh/develop/03.image_classification/index.cn.html)
1. [词向量](http://www.paddlepaddle.org/documentation/book/zh/develop/04.word2vec/index.cn.html)
1. [个性化推荐](http://www.paddlepaddle.org/documentation/book/zh/develop/05.recommender_system/index.cn.html)
1. [情感分析](http://www.paddlepaddle.org/documentation/book/zh/develop/06.understand_sentiment/index.cn.html)
1. [语义角色标注](http://www.paddlepaddle.org/documentation/book/zh/develop/07.label_semantic_roles/index.cn.html)
1. [机器翻译](http://www.paddlepaddle.org/documentation/book/zh/develop/08.machine_translation/index.cn.html)

更多学习内容请访问PaddlePaddle[视频课堂](http://bit.baidu.com/Course/datalist/column/117.html)。

## 运行这本书

您现在在看的这本书是一本“交互式”电子书 —— 每一章都可以运行在一个Jupyter Notebook里。

我们把Jupyter、PaddlePaddle、以及各种被依赖的软件都打包进一个Docker image了。所以您不需要自己来安装各种软件，只需要安装Docker即可。对于各种Linux发行版，请参考 https://www.docker.com 。如果您使用[Windows](https://www.docker.com/docker-windows)或者[Mac](https://www.docker.com/docker-mac)，可以考虑[给Docker更多内存和CPU资源](http://stackoverflow.com/a/39720010/724872)。

只需要在命令行窗口里运行：

```bash
docker run -d -p 8888:8888 paddlepaddle/book
```

会从DockerHub.com下载和运行本书的Docker image。阅读和在线编辑本书请在浏览器里访问 http://localhost:8888 。

如果您访问DockerHub.com很慢，可以试试我们的另一个镜像hub.baidubce.com：

```bash
docker run -d -p 8888:8888 hub.baidubce.com/paddlepaddle/book
```

### 使用GPU训练

本书默认使用CPU训练，若是要使用GPU训练，使用步骤会稍有变化。为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。请先安装nvidia-docker，之后请运行：

```bash
nvidia-docker run -d -p 8888:8888 paddlepaddle/book:latest-gpu
```

或者使用国内的镜像请运行：

```bash
nvidia-docker run -d -p 8888:8888 hub.baidubce.com/paddlepaddle/book:latest-gpu
```

还需要将以下代码
```python
use_cuda = False
```

改成：
```python
use_cuda = True
```


## 贡献内容

您要是能贡献新的章节那就太好了！请发Pull Requests把您写的章节加入到`/pending`下面的一个子目录里。当这一章稳定下来，我们一起把您的目录挪到根目录。

为了写作、运行、调试，您需要安装Python 2.x和Go >1.5, 并可以用[脚本程序](https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh)来生成新的Docker image。

**Note:** We also provide [English Readme](https://github.com/PaddlePaddle/book/blob/develop/README.md) for PaddlePaddle book.


<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">本教程</span> 由 <a xmlns:cc="http://creativecommons.org/ns#" href="http://www.paddlepaddle.org/" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a> 创作，采用 <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">知识共享 署名-相同方式共享 4.0 国际 许可协议</a>进行许可。
