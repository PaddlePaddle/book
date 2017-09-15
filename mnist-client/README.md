# MNIST classification by PaddlePaddle

![screencast](https://cloud.githubusercontent.com/assets/80381/11339453/f04f885e-923c-11e5-8845-33c16978c54d.gif)

## Usage

This MNIST classification demo consists of two parts: a PaddlePaddle
inference server and a Javascript front end. We will start them
separately.

We will use Docker to run the demo, if you are not familiar with
Docker, please checkout
this
[tutorial](https://github.com/PaddlePaddle/Paddle/wiki/TLDR-for-new-docker-user).

### Start the Inference Server

The inference server can be used to inference any model trained by
PaddlePaddle. Please see [here](../serve/README.md) for more details.

1. Download the MNIST inference model topylogy and parameters to the
   current working directory.

    ```bash
    wget https://s3.us-east-2.amazonaws.com/models.paddlepaddle/end-to-end-mnist/inference_topology.pkl
    wget https://s3.us-east-2.amazonaws.com/models.paddlepaddle/end-to-end-mnist/param.tar
    ```

1. Run following command to start the inference server:

    ```bash
    docker run --name paddle_serve -v `pwd`:/data -d -p 8000:80 -e WITH_GPU=0 paddlepaddle/book:serve
    ```

    The above command will mount the current working directory to the
    `/data` directory inside the docker container. The inference
    server will load the model topology and parameters that we just
    downloaded from there.

    After you are done with the demo, you can run `docker stop
    paddle_serve` to stop this docker container.

### Start the Front End

1. Run the following command
   ```bash
   docker run -it -p 5000:5000 -e BACKEND_URL=http://localhost:8000/ paddlepaddle/book:mnist
   ```

   `BACKEND_URL` in the above command specifies the inference server
   endpoint. If you started the inference server on another machine,
   or want to visit the front end remotely, you may want to change its
   value.

1. Visit http://localhost:5000 and you will see the PaddlePaddle MNIST demo.


## Build

We have already prepared the pre-built docker image
`paddlepaddle/book:mnist`, here is the command if you want to build
the docker image again.

```bash
docker build -t paddlepaddle/book:mnist .
```


## Acknowledgement

Thanks to the great project https://github.com/sugyan/tensorflow-mnist
. Most of the code in this project comes from there.
