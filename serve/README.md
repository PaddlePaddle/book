# PaddlePaddle Serving Example


## Build

    $ docker build -t serve .

## Run

    $ docker run -v `pwd`:/data -it -p 8000:80 -e WITH_GPU=0 paddlepaddle/book:serve
    $ curl -H "Content-Type: application/json" -X POST -d '{"img":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}' http://localhost:8000/


## How to save PaddlePaddle model

Neural network model in PaddlePaddle contains two parts, the parameter, and the topology.

Paddle training scripts contain the neural network topology, which is representing by layers. For example,

```python
img = paddle.layer.data(name="img", type=paddle.data_type.dense_vector(784))
hidden = fc_layer(input=type, size=200)
prediction = fc_layer(input=hidden, size=10, act=paddle.activation.Softmax())
```

The parameter instance is created by topology and updated by the `train` method.

```python
...
params = paddle.parameters.create(cost)
...
trainer = paddle.trainer.SGD(cost=cost, parameters=params)
...
```

PaddlePaddle stores the topology and parameter separately.

1. To serialize a topology, we need to create a topology instance explicitly by the outputs of the neural network. Then, invoke `serialize_for_inference` method. The example code is

  ```python
  # Save the inference topology to protobuf.
  inference_topology = paddle.topology.Topology(layers=prediction)
  with open("inference_topology.pkl", 'wb') as f:
      inference_topology.serialize_for_inference(f)
  ```

2. To save a parameter, we need to invoke `to_tar` method in Parameter class. The example code is,

  ```python
  with open('param.tar', 'w') as f:
            params.to_tar(f)
  ```

 After we serialize the parameter and topology to two files, we could use that two files to set up an inference server.


## How to set up an inference server

...


## What is the data format of inference server

The inference server will handle a post request on uri `/`. The contant type of the request and response is json. You need to manually add `Content-Type` request header as `Content-Type: application/json`.

The request json object is a single json object, which key is the layer name of input data. The value of that object is decided by data type.

There are tweleve data types are supported by PaddePaddle, and they are organized in a matrix.

| | plain | a sequence | a sequence of sequence |
| --- | --- | --- | ---|
| dense | [ f, f, f, f, ... ] | [ [f, f, f, ...], [f, f, f, ...]] | [[[f, f, ...], [f, f, ...]], [[f, f, ...], [f, f, ...]], ...] |
| integer | i | [i, i, ...] | [[i, i, ...], [i, i, ...], ...] |
| sparse | [i, i, ...] | [[i, i, ...], [i, i, ...], ...] | [[[i, i, ...], [i, i, ...], ...], [[i, i, ...], [i, i, ...], ...], ...] |
| sparse | [[i, f], [i, f], ... ] | [[[i, f], [i, f], ... ], ...] | [[[[i, f], [i, f], ... ], ...], ...]

In that table, `i` stands for a `int` value and `f` stands for a `float` value.

What `data_type` should be used is decided by the training topology. For example,

* For image data, they are usually a plain dense vector, we flatten the image into a vector. The pixels of that image are usually normalized in `[-1.0, 1.0]` or `[0.0, 1.0]`(it depends on each neural network.).

    ```text
    +-------+
   |243 241|
   |139 211| +---->[0.95, 0.95, 0.54, 0.82]
   +-------+
    ```
* For text data, each word of that text is represented by a integer. The association map between word and integer is decided by the training process. A sentence is represented by a list of integer.

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

The `code` and `message` represent the status of the request. The `data` are the outputs of the neural network; they could be a probability of each class, could be the IDs of output sentence, and so on.
