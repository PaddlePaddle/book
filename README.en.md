# Deep Learning with PaddlePaddle

1. [Fit a Line](http://book.paddlepaddle.org/fit_a_line/index.en.html)
1. [Recognize Digits](http://book.paddlepaddle.org/recognize_digits/index.en.html)
1. [Image Classification](http://book.paddlepaddle.org/image_classification/index.en.html)
1. [Word to Vector](http://book.paddlepaddle.org/word2vec/index.en.html)
1. [Understand Sentiment](http://book.paddlepaddle.org/understand_sentiment/index.en.html)
1. [Label Semantic Roles](http://book.paddlepaddle.org/label_semantic_roles/index.en.html)
1. [Machine Translation](http://book.paddlepaddle.org/machine_translation/index.en.html)
1. [Recommender System](http://book.paddlepaddle.org/recommender_system/index.en.html)

# Reading Using Docker
You can read this book and test the notebookfile using Docker.  If you can access dockerhub, run it as  
```bash
docker run -d -p 8888:8888 paddlepaddle/book
```

Or if you are in china mainland, you can use  
```bash
docker run -d -p 8888:8888 docker.paddlepaddle.org/book
```

Then open the url http://ip:8888/, such as:  
```
http://localhost:8888/
```

# How to build book's docker image
1.prepare your docker environment, make sure you can run docker command  

```bash
docker
```

2.get go [here](https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz)

3.get book source code and build it  

```bash
git clone https://github.com/PaddlePaddle/book.git
cd book/
.tools/build_docker.sh
```

# How to get paddlepaddle docker image
This book's docker image depends on paddlepaddle docker image. Paddlepaddle's document is [here](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/docker)


This tutorial is contributed by <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a>, and licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
