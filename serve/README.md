# Inference Server Example

The inference server can be used to perform inference on any model trained on
PaddlePaddle. It provides an HTTP endpoint.

## Run

The inference server reads a trained model (a topology file and a
parameter file) and serves HTTP request at port `8000`. Because models
differ in the numbers and types of inputs, **the HTTP API will differ
slightly for each model,** please see [HTTP API](#http-api) for the
API spec,
and
[here](https://github.com/PaddlePaddle/book/wiki/Using-Pre-trained-Models) for
the request examples of different models that illustrate the
difference.

We will first show how to obtain the PaddlePaddle model, and then how
to start the server.

We will use Docker to run the demo, if you are not familiar with
Docker, please checkout
this
[TLDR](https://github.com/PaddlePaddle/Paddle/wiki/Docker-for-Beginners).

### Obtain the PaddlePaddle Model

A neural network model in PaddlePaddle contains two parts: the
**parameter** and the **topology**.

A PaddlePaddle training script contains the neural network topology,
which is represented by layers. For example,

```python
img = paddle.layer.data(name="img", type=paddle.data_type.dense_vector(784))
hidden = fc_layer(input=type, size=200)
prediction = fc_layer(input=hidden, size=10, act=paddle.activation.Softmax())
```

The parameter instance is created by the topology and updated by the
`train` method.

```python
...
params = paddle.parameters.create(cost)
...
trainer = paddle.trainer.SGD(cost=cost, parameters=params)
...
```

PaddlePaddle stores the topology and parameter separately.

1. To serialize a topology, we need to create a topology instance
   explicitly by the outputs of the neural network. Then, invoke
   `serialize_for_inference` method.

  ```python
  # Save the inference topology to protobuf.
  inference_topology = paddle.topology.Topology(layers=prediction)
  with open("inference_topology.pkl", 'wb') as f:
      inference_topology.serialize_for_inference(f)
  ```

2. To save a parameter, we need to invoke `save_parameter_to_tar` method of
  `trainer`.

  ```python
  with open('param.tar', 'w') as f:
      trainer.save_parameter_to_tar(f)
  ```

 After serializing the parameter and topology into two files, we could
 use them to set up an inference server.

 For a working example, please see [train.py](https://github.com/reyoung/paddle_mnist_v2_demo/blob/master/train.py).


### Start the Server

Make sure the `inference_topology.pkl` and `param.tar` mentioned in
the last section are in your current working directory, and run the
command:

```bash
docker run --name paddle_serve -v `pwd`:/data -d -p 8000:80 -e WITH_GPU=0 paddlepaddle/book:serve
```

The above command will mount the current working directory to the
`/data/` directory inside the docker container. The inference server
will load the model topology and parameters that we just created from
there.

To run the inference server with GPU support, please make sure you have
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
first, and run:

```bash
nvidia-docker run --name paddle_serve -v `pwd`:/data -d -p 8000:80 -e WITH_GPU=1 paddlepaddle/book:serve-gpu
```

this command will start a server on port `8000`.

After you are done with the demo, you can run `docker stop
paddle_serve` to stop this docker container.

## HTTP API

The inference server will handle HTTP POST request on path `/`. The
content type of the request and response is json. You need to manually
add `Content-Type` request header as `Content-Type: application/json`.

The request json object is a single json dictionay object, whose key
is the layer name of input data. The type of the corresponding value
is decided by the data type. For most cases the corresponding value
will be a list of floats. For completeness, we will list all data types
below:

There are twelve data types supported by PaddePaddle:

| | plain | a sequence | a sequence of sequence |
| --- | --- | --- | ---|
| dense | [ f, f, f, f, ... ] | [ [f, f, f, ...], [f, f, f, ...]] | [[[f, f, ...], [f, f, ...]], [[f, f, ...], [f, f, ...]], ...] |
| integer | i | [i, i, ...] | [[i, i, ...], [i, i, ...], ...] |
| sparse | [i, i, ...] | [[i, i, ...], [i, i, ...], ...] | [[[i, i, ...], [i, i, ...], ...], [[i, i, ...], [i, i, ...], ...], ...] |
| sparse | [[i, f], [i, f], ... ] | [[[i, f], [i, f], ... ], ...] | [[[[i, f], [i, f], ... ], ...], ...]

In the table, `i` stands for a `int` value and `f` stands for a
`float` value.

What `data_type` should be used is decided by the training
topology. For example,

* For image data, they are usually a plain dense vector, we flatten
  the image into a vector. The pixel values of that image are usually
  normalized in `[-1.0, 1.0]` or `[0.0, 1.0]`(depends on each neural
  network).

    ```text
    +-------+
   |243 241|
   |139 211| +---->[0.95, 0.95, 0.54, 0.82]
   +-------+
    ```

* For text data, each word of that text is represented by an
  integer. The association map between word and integer is decided by
  the training process. A sentence is represented by a list of
  integer.

   ```text
    I am good .
        +
        |
        v
   23 942 402 19  +----->  [23, 942, 402, 19]
   ```

A sample request data of a `4x4` image and a sentence could be

```json
{
    "img": [
        0.95,
        0.95,
        0.54,
        0.82
    ],
    "sentence": [
        23,
        942,
        402,
        19
    ]
}
```

The response is a json object, too. The example of return data are:

```json
{
  "code": 0,
  "data": [
    [
      0.10060056298971176,
      0.057179879397153854,
      0.1453431099653244,
      0.15825574100017548,
      0.04464773088693619,
      0.1566203236579895,
      0.05657859891653061,
      0.12077419459819794,
      0.08073269575834274,
      0.07926714420318604
    ]
  ],
  "message": "success"
}
```

Here, `code` and `message` represent the status of the request.
`data` corresponds to the outputs of the neural network; they could be a
probability of each class, could be the IDs of output sentence, and so
on.

## MNIST Demo Client

If you have trained an model with [train.py](https://github.com/reyoung/paddle_mnist_v2_demo/blob/master/train.py) and
start a inference server. Then you can use this [client](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits/client/client.py) to test if it works right.

## Build

We have already prepared the pre-built docker image
`paddlepaddle/book:serve`, here is the command if you want to build
the docker image again.

```bash
docker build -t paddlepaddle/book:serve .
docker build -t paddlepaddle/book:serve-gpu -f Dockerfile.gpu .
```
