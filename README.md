# 深度学习入门

[![Build Status](https://travis-ci.org/PaddlePaddle/book.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/book)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://book.paddlepaddle.org/)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://book.paddlepaddle.org/index.en.html)

1. [新手入门](http://book.paddlepaddle.org/01.fit_a_line)
1. [识别数字](http://book.paddlepaddle.org/02.recognize_digits)
1. [图像分类](http://book.paddlepaddle.org/03.image_classification)
1. [词向量](http://book.paddlepaddle.org/04.word2vec)
1. [情感分析](http://book.paddlepaddle.org/05.understand_sentiment)
1. [语义角色标注](http://book.paddlepaddle.org/06.label_semantic_roles)
1. [机器翻译](http://book.paddlepaddle.org/07.machine_translation)
1. [个性化推荐](http://book.paddlepaddle.org/08.recommender_system)

## 运行这本书

您现在在看的这本书是一本“交互式”电子书 —— 每一章都可以运行在一个
Jupyter Notebook 里。

我们把 Jupyter、PaddlePaddle、以及各种被依赖的软件都打包进一个 Docker
image 了。所以您不需要自己来安装各种软件，只需要安装 Docker 即可。如果
您使用 Windows，可以参
考[这里](https://www.docker.com/docker-windows)。如果您使用 Mac，可以
参考[这里](https://www.docker.com/docker-mac)。 对于各种 Linux 发行版，
请参考https://www.docker.com 。如果您使用 Windows 或者 Mac，可以通过如
下方法给 Docker 更多内存和CPU资源
(http://stackoverflow.com/a/39720010/724872)。

只需要在命令行窗口里运行：

```bash
docker run -d -p 8888:8888 paddlepaddle/book
```

这个命令会从 DockerHub.com 下载本书的 Docker image 并且运行之。请在浏
览器里访问 http://localhost:8888 即可阅读和在线编辑本书。

如果您访问 DockerHub.com 很慢，可以试试我们的另一个镜像
docker.paddlepaddle.org：

```bash
docker run -d -p 8888:8888 docker.paddlepaddle.org/book
```

## 贡献内容

您要是能贡献新的章节那就太好了！请发 Pull Requests 把您写的章节加入到
`/pending` 下面的一个子目录里。当这一章稳定下来，我们一起把您的目录挪
到根目录。

为了写作、运行、调试，您需要安装 Python 2.x, Go >1.5. 你可以用这
个[脚本程序](https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh)来
生成 Docker image。


**Note:** We also provide [English Readme](https://github.com/PaddlePaddle/book/blob/develop/README.en.md) for PaddlePaddle book.


<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">本教程</span> 由 <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a> 创作，采用 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享 署名-非商业性使用-相同方式共享 4.0 国际 许可协议</a>进行许可。
