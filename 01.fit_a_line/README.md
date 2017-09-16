# Linear Regression
Let us begin the tutorial with a classical problem called Linear Regression \[[1](#References)\]. In this chapter, we will train a model from a realistic dataset to predict home prices. Some important concepts in Machine Learning will be covered through this example.

The source code for this tutorial lives on [book/fit_a_line](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line). For instructions on getting started with PaddlePaddle, see [PaddlePaddle installation guide](https://github.com/PaddlePaddle/book/blob/develop/README.md#running-the-book).

## Problem Setup
Suppose we have a dataset of $n$ real estate properties. These real estate properties will be referred to as *homes* in this chapter for clarity.

Each home is associated with $d$ attributes. These attributes describe characteristics such the number of rooms in a home, the number of schools in a neighborhood, or the traffic condition nearby.

In this problem setup, the attribute $x_{i,j}$ denotes the $j$th characteristic of the $i$th home. In addition, $y_i$ denotes the price of the $i$th home. Our task is to predict $y_i$ given a set of attributes $\{x_{i,1}, ..., x_{i,d}\}$. Let's assume that the price of a home is a linear combination of all its attributes, namely:

$$y_i = \omega_1x_{i,1} + \omega_2x_{i,2} + \ldots + \omega_dx_{i,d} + b,  i=1,\ldots,n$$

where $\vec{\omega}$ and $b$ are the model parameters we want to estimate. Once we learn these, we'll be able to predict the price of any home, given the attributes associated with it. This model is called **Linear Regression**. In other words, we regress a value against several other values linearly. However in reality, a linear model is often too simplistic to capture the real relationships between variables. Yet, because Linear Regression is easy to train and analyze, it has been applied to a large number of real problems. As a result, it is an important and recurring topic in most classic Statistical and Machine Learning textbooks \[[2,3,4](#References)\].

## Results Demonstration
Let's take a look at the results our model provides. We used this dataset [UCI Housing Data Set](https://archive.ics.uci.edu/ml/datasets/Housing) to train a linear model to predict housing prices in Boston. The figure below shows the price predictions the model makes for several homes. The $X$-axis represents the median prices of similar homes within a bin, while the $Y$-axis represents the value predicted by our linear model. The dotted line represents points where $X=Y$. The more precise the model's predictions, the closer the point is to the dotted line.
<p align="center">
    <img src = "image/predictions_en.png" width=400><br/>
    Figure 1. Predicted Value Versus Actual Value
</p>

## Model Overview

### Model Definition

In the UCI Housing Data Set, there are 13 home attributes $\{x_{i,j}\}$ that are related to median home price $y_i$, which we wish to foretell. Thus, our model can be written as:

$$\hat{Y} = \omega_1X_{1} + \omega_2X_{2} + \ldots + \omega_{13}X_{13} + b$$

where $\hat{Y}$ is the predicted value contrasted with the actual value $Y$. The model learns the parameters $\omega_1, \ldots, \omega_{13}, b$, where the entries of $\vec{\omega}$ are **weights** and $b$ is **bias**.

Now we need an objective to optimize so the learned parameters can make $\hat{Y}$ match $Y$ as closely as possible. This is where the [Loss Function (Cost Function)](https://en.wikipedia.org/wiki/Loss_function) comes in handy. A loss function, given any pair of the actual value $y_i$ and the predicted value $\hat{y_i}$, outputs a non-negative value that reflects the magnitude of the model's error.

For Linear Regression, the most common loss function is [Mean Square Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) which takes the following form:

$$MSE=\frac{1}{n}\sum_{i=1}^{n}{(\hat{Y_i}-Y_i)}^2$$

That is, for a dataset of size $n$, MSE is the average value of the prediction square errors.

### Training

After setting up our model, there are several major steps to go through to train it:
1. Initialize the parameters, including the weights $\vec{\omega}$ and the bias $b$. For example, we can set their mean values as $0$s, and their standard deviations as $1$s.
2. Feedforward. Evaluate the network output and compute the corresponding loss.
3. [Backpropagate](https://en.wikipedia.org/wiki/Backpropagation) the errors. The errors will be propagated from the output layer back to the input layer. During this process, the model parameters will be updated along with the corresponding errors.
4. Repeat steps 2 and 3 until the loss is below a predefined threshold or the maximum number of repeats is reached.

## Dataset

### Python Dataset Modules

Our program starts with importing necessary packages:

```python
import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
```

We encapsulated the [UCI Housing Data Set](https://archive.ics.uci.edu/ml/datasets/Housing) in our Python module `uci_housing`.  This module can:

1. download the dataset to `~/.cache/paddle/dataset/uci_housing/housing.data` and
2. [preprocess](#preprocessing) the dataset.

### An Introduction of the Dataset

The UCI housing dataset has 506 instances. Each instance describes the attributes of a house in surburban Boston.  The attributes are explained below:

| Attribute Name | Characteristic | Data Type |
| ------| ------ | ------ |
| CRIM | per capita crime rate by town | Continuous|
| ZN | proportion of residential land zoned for lots over 25,000 sq.ft. | Continuous |
| INDUS | proportion of non-retail business acres per town | Continuous |
| CHAS | Charles River dummy variable | Discrete, 1 if tract bounds river; 0 otherwise|
| NOX | nitric oxides concentration (parts per 10 million) | Continuous |
| RM | average number of rooms per dwelling | Continuous |
| AGE | proportion of owner-occupied units built prior to 1940 | Continuous |
| DIS | weighted distances to five Boston employment centres | Continuous |
| RAD | index of accessibility to radial highways | Continuous |
| TAX | full-value property-tax rate per $10,000 | Continuous |
| PTRATIO | pupil-teacher ratio by town | Continuous |
| B | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | Continuous |
| LSTAT | % lower status of the population | Continuous |
| MEDV | Median price value of owner-occupied homes in $1000's | Continuous |

### Preprocessing
#### Continuous and Discrete Data
We've defined a feature vector of length 13 for each home, where each entry corresponds to an attribute. Our first observation is that, among the 13 dimensions, there are 12 continuous ones and 1 discrete.

Note that although a discrete value can also be written as a number, its meaning drastically differs from a continuous value's. The linear difference between two discrete values has no meaning. For example, suppose $0$, $1$, and $2$ were used to represent the colors *Red*, *Green*, and *Blue* respectively. Judging from the numeric representation of these colors, *Red* would differ more from *Blue* than it does from *Green*. Yet in actuality, the difference between *Blue* and *Red* is not greater than the difference between *Green* and *Red*. Therefore, when handling a discrete feature that has $d$ possible values, we usually convert it to $d$ new features where each feature takes a binary value ($0$ or $1$) indicating the absence or presence of the original value. Alternatively, the discrete features can also be mapped onto a continuous multi-dimensional vector through an embedding table. For our problem here, because CHAS itself is a binary discrete value, no preprocessing is necessary.

#### Feature Normalization
We can also observe a significant difference among the value ranges of the 13 features (Figure 2). For instance, the values of feature *B* fall in $[0.32, 396.90]$, whereas those of feature *NOX* have a range of $[0.3850, 0.8170]$. An effective optimization would require data normalization, which scales the values of each feature into roughly the same range, perhaps $[-0.5, 0.5]$. Here, we've adopted the popular normalization technique of substracting the mean value from the feature value, then dividing the result by the original range's width.

[Feature Normalization](https://en.wikipedia.org/wiki/Feature_scaling) (Feature Scaling) can be necessary for three reasons:
- A value range that is too large or too small can cause floating number overflow or underflow during computation.
- Different value ranges might result in varying *importances* of different features to the model (at least in the beginning of the training process). This assumption about the data is often unreasonable, making optimization difficult and increasing training time.
- Many machine learning techniques or models (e.g., *L1/L2 regularization* and *Vector Space Model*) assume that all the features have roughly zero means and that their value ranges are similar.

<p align="center">
    <img src = "image/ranges_en.png" width=550><br/>
    Figure 2. The value ranges of the features
</p>

#### Prepare Training and Test Sets
To prepare, we've split the dataset in two parts: One for adjusting the model parameters, namely for model training, and the other for model testing. The model error on the former is called the **training error**, and the error on the latter is called the **test error**. Our goal in training a model is to find the statistical dependency between outputs and inputs, so that we may predict the outputs of new given inputs. Therefore, the test error reflects the performance of the model more accurately than the training error. Two factors determine the ratio of the training set to the test set: 1) More training data will decrease the variance of the parameter estimation, yielding more reliable models; 2) More test data will decrease the variance of the test error, yielding more reliable test errors. One standard split ratio is $8:2$.


When training complex models, we usually have one more split: the validation set. Complex models usually have [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_optimization) that must be set before training, such as the number of layers in the network. Because hyperparameters are not part of the model parameters, they cannot be trained using the same loss function. Instead, we will try several sets of hyperparameters to train several models and cross-validate them on the validation set to pick the best one. Lastly, the selected, trained model is tested on the test set. Because our model is relatively simple, we will omit this validation process.


## Training

`fit_a_line/trainer.py` demonstrates the training using [PaddlePaddle](http://paddlepaddle.org).

### Initialize PaddlePaddle

```python
paddle.init(use_gpu=False, trainer_count=1)
```

### Model Configuration

Linear regression is essentially a fully-connected layer with linear activation:

```python
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
y_predict = paddle.layer.fc(input=x,
                                size=1,
                                act=paddle.activation.Linear())
y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
cost = paddle.layer.square_error_cost(input=y_predict, label=y)
```

### Save Topology

```python
# Save the inference topology to protobuf.
inference_topology = paddle.topology.Topology(layers=y_predict)
with open("inference_topology.pkl", 'wb') as f:
    inference_topology.serialize_for_inference(f)
```


### Create Parameters

```python
parameters = paddle.parameters.create(cost)
```

### Create Trainer

```python
optimizer = paddle.optimizer.Momentum(momentum=0)

trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)
```

### Feeding Data

PaddlePaddle provides the
[reader mechanism](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/reader)
for loadinng training data. A reader may return multiple columns, and we need a Python dictionary to specify the mapping from column index to data layers.

```python
feeding={'x': 0, 'y': 1}
```

Moreover, an event handler is provided to print the training progress:

```python
# event_handler to print training and testing info
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f" % (
                event.pass_id, event.batch_id, event.cost)

    if isinstance(event, paddle.event.EndPass):
        result = trainer.test(
            reader=paddle.batch(
                uci_housing.test(), batch_size=2),
            feeding=feeding)
        print "Test %d, Cost %f" % (event.pass_id, result.cost)
```

```python
# event_handler to print training and testing info
from paddle.v2.plot import Ploter

train_title = "Train cost"
test_title = "Test cost"
plot_cost = Ploter(train_title, test_title)

step = 0

def event_handler_plot(event):
    global step
    if isinstance(event, paddle.event.EndIteration):
        if step % 10 == 0:  # every 10 batches, record a train cost
            plot_cost.append(train_title, step, event.cost)

        if step % 100 == 0: # every 100 batches, record a test cost
            result = trainer.test(
                reader=paddle.batch(
                    uci_housing.test(), batch_size=2),
                feeding=feeding)
            plot_cost.append(test_title, step, result.cost)

        if step % 100 == 0: # every 100 batches, update cost plot
            plot_cost.plot()

        step += 1

    if isinstance(event, paddle.event.EndPass):
        if event.pass_id % 10 == 0:
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
```

### Start Training

```python
trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            uci_housing.train(), buf_size=500),
        batch_size=2),
    feeding=feeding,
    event_handler=event_handler_plot,
    num_passes=30)
```

![png](./image/train_and_test.png)

### Apply model

#### 1. generate testing data

```python
test_data_creator = paddle.dataset.uci_housing.test()
test_data = []
test_label = []

for item in test_data_creator():
    test_data.append((item[0],))
    test_label.append(item[1])
    if len(test_data) == 5:
        break
```

#### 2. inference

```python
# load parameters from tar file.
# users can remove the comments and change the model name
# with open('params_pass_20.tar', 'r') as f:
#     parameters = paddle.parameters.Parameters.from_tar(f)

probs = paddle.infer(
    output_layer=y_predict, parameters=parameters, input=test_data)

for i in xrange(len(probs)):
    print "label=" + str(test_label[i][0]) + ", predict=" + str(probs[i][0])
```

## Summary
This chapter introduces *Linear Regression* and explains how to train and test this model with PaddlePaddle using the UCI Housing Data Set. Because a large number of more complex models and techniques are derived from linear regression, it is important to understand its underlying theory and various limitations.


## References
1. https://en.wikipedia.org/wiki/Linear_regression
2. Friedman J, Hastie T, Tibshirani R. The elements of statistical learning[M]. Springer, Berlin: Springer series in statistics, 2001.
3. Murphy K P. Machine learning: a probabilistic perspective[M]. MIT press, 2012.
4. Bishop C M. Pattern recognition[J]. Machine Learning, 2006, 128.

<br/>
This tutorial is contributed by <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a>, and licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
