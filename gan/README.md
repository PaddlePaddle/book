# 对抗式生成网络

## 背景介绍

本章我们介绍对抗式生成网络，也称为Generative Adversarial Network (GAN) \[[1](#参考文献)\]。GAN的核心思想是，为了更好地训练一个生成式神经元网络模型（generative model），我们引入一个分类神经元网络模型来构造优化目标函数。实验证明，在图像自动生成、图像去噪、和确实图像不全等应用里，这种方法可以更容易地得到一个能更好逼近训练数据分布的生成式模型。

到目前为止，大部分在应用中取得好效果的神经元网络模型都是有监督训练（supervised learning）的判别式模型（discriminative models），包括图像识别中使用的convolutional networks和在语音识别中使用的connectionist temporal classification (CTC) networks。在这些例子里，训练数据 X 都是带有标签 y 的——每张图片附带了一个或者多个tag，每段语音附带了一段对应的文本；而模型的输入是 X，输出是 y，表示从X到y的映射函数 y=f(X)。

和判别式神经元网络模型相对的一类模型是生成式模型（generative models）。它们通常是通过非监督训练（unsupervised learning）来得到的。这类模型的训练数据里只有 X，没有y。训练的目标是希望模型能包含训练数据的分布，从而可以从训练好的模型里产生出新的、在训练数据里没有出现过的数据 x'。

本文里，我们介绍如何训练一个产生式神经元网络模型，它的输入是一个随机生成的向量（相当于不需要任何有意义的输入），而输出是一幅图像，其中有一个数字。换句话说，我们训练一个会写字（阿拉伯数字）的神经元网络模型。它“写”的一些数字如下图：

<p align="center">
    <img src="./mnist_sample.png" width="300" height="300"><br/>
    图1. GAN生成的MNIST例图
</p>

现实中成功使用的生成式神经元网络模型往往接受有意义的输入。比如可能接受一幅低分辨率的图像，输出对应的高分辨率图像。这过程实际上是从大量数据学习得到模型，或者说归纳得到知识，然后用这些知识来补足图像的分辨率。

## 传统训练方式和对抗式训练

因为神经元网络是一个有向图，总是有输入和输出的。当我们用无监督学习方式来训练一个神经元网络，用于描述训练数据分布的时候，一个通常的学习目标是估计一组参数，使得输出和输入很接近 —— 或者说输入是什么输出就是什么。很多早期的生成式神经元网络模型，包括有限制波尔茨曼机（restricted Boltzmann machine，RBM）和 autoencoder 都是这么训练的。这种情况下优化目标经常是最小化输出和输入的差别。

对抗式训练里，我们用一个判别式模型 D 辅助构造优化目标函数，来训练一个生成式模型 G。具体训练流程是不断交替执行如下两步：

1. 更新模型 D：
   1. 固定G的参数不变，对于一组随机输入，得到一组（产生式）输出，$X_f$，并且将其label成“假”。
   1. 从训练数据 X 采样一组 $X_r$，并且label为“真”。
   1. 用这两组数据更新模型 D，从而使D能够分辨G产生的数据和真实训练数据。

1. 更新模型 G：
   1. 把G的输出和D的输入连接起来，得到一个网路。
   1. 给G一组随机输入，期待G的输出让D认为像是“真”的。
   1. 在D的输出端，优化目标是通过更新G的参数来最小化D的输出和“真”的差别。

上述方法实际上在优化如下目标：

$$\min_G \max_D \frac{1}{N}\sum_{i=1}^N[\log D(x^i) + \log(1-D(G(z^i)))]$$

其中$x$是真实数据，$z$是随机产生的输入，$N$是训练数据的数量。这个损失函数的意思是：真实数据被分类为真的概率加上伪数据被分类为假的概率。因为上述两步交替优化G生成的结果的仿真程度（看起来像x）和D分辨G的生成结果和x的能力，所以这个方法被称为对抗（adversarial）方法。

在最早的对抗式生成网络的论文中，生成器和分类器用的都是全联接层，所以没有办法很好的生成图片数据，也没有办法做的很深。所以在随后的论文中，人们提出了深度卷积对抗式生成网络（deep convolutional generative adversarial network or DCGAN）\[[2](#参考文献)\]。在DCGAN中，生成器 G 是由多个卷积转置层（transposed convolution）组成的，这样可以用更少的参数来生成质量更高的图片。具体网络结果可参见图3。

<p align="center">
    <img src="./dcgan.png" width="700" height="300"><br/>
    图3. DCGAN生成器模型结构
    <a href="https://arxiv.org/pdf/1511.06434v2.pdf">figure credit</a>
</p>


## 数据准备

todo(yi): from here on

### 数据介绍与下载
这章会用到两种数据，一种是简单的人造数据，一种是图片。

人造数据是二维均匀分布，由下面的代码生成:

```python
# synthesize 2-D uniform data in gan_trainer.py:114
def load_uniform_data():
    data = numpy.random.rand(1000000, 2).astype('float32')
    return data
```

图片数据是MNIST手写数字，可由下面的代码下载：

```bash
$cd data/
$./get_mnist_data.sh
```

另一种更真实的图片数据是Cifar-10，可由下面的代码下载：

```bash
$cd data/
$./download_cifar.sh
```

## 模型配置说明
由于对抗式生产网络涉及到多个神经网络，所以必须用paddle Python API来训练。下面的介绍也可以部分的拿来当作paddle Python API的使用说明。

### 模型结构
在文件gan_conf.py当中我们定义了三个网络, **generator_training**, **discriminator_training** and **generator**. 和前文提到的模型结构的关系是：**discriminator_training** 是分类器，**generator** 是生成器，**generator_training** 是生成器加分类器因为训练生成器时需要用到分类器提供目标函数。这个对应关系在下面这段代码中定义：

```python
if is_generator_training:
    noise = data_layer(name="noise", size=noise_dim)
    sample = generator(noise)

if is_discriminator_training:
    sample = data_layer(name="sample", size=sample_dim)

if is_generator_training or is_discriminator_training:
    label = data_layer(name="label", size=1)
    prob = discriminator(sample)
    cost = cross_entropy(input=prob, label=label)
    classification_error_evaluator(
        input=prob, label=label, name=mode + '_error')
    outputs(cost)

if is_generator:
    noise = data_layer(name="noise", size=noise_dim)
    outputs(generator(noise))
```

为了能够训练在gan_conf.py中定义的网络，我们需要如下几个步骤：初始化Paddle环境，解析设置，由设置创造GradientMachine以及由GradientMachine创造trainer。这几步分别由下面几段代码实现：

```python
import py_paddle.swig_paddle as api
# init paddle environment
api.initPaddle('--use_gpu=' + use_gpu, '--dot_period=10',
               '--log_period=100', '--gpu_id=' + args.gpu_id,
               '--save_dir=' + "./%s_params/" % data_source)

# Parse config
gen_conf = parse_config(conf, "mode=generator_training,data=" + data_source)
dis_conf = parse_config(conf, "mode=discriminator_training,data=" + data_source)
generator_conf = parse_config(conf, "mode=generator,data=" + data_source)

# Create GradientMachine
dis_training_machine = api.GradientMachine.createFromConfigProto(
dis_conf.model_config)
gen_training_machine = api.GradientMachine.createFromConfigProto(
gen_conf.model_config)
generator_machine = api.GradientMachine.createFromConfigProto(
generator_conf.model_config)

# Create trainer
dis_trainer = api.Trainer.create(dis_conf, dis_training_machine)
gen_trainer = api.Trainer.create(gen_conf, gen_training_machine)
```

为了能够平衡生成器和分类器之间的能力，我们依据它们各自的损失函数的大小来决定训练对象，即我们选择训练那个损失函数更大的网络。损失函数的值可以通过GradientMachine的forward pass来计算。

```python
def get_training_loss(training_machine, inputs):
    outputs = api.Arguments.createArguments(0)
    training_machine.forward(inputs, outputs, api.PASS_TEST)
    loss = outputs.getSlotValue(0).copyToNumpyMat()
    return numpy.mean(loss)
```

每当训练完一个网络，我们需要和其他几个网络同步互相分享的参数值。下面的代码展示了其中一个例子：

```python
# Train the gen_training
gen_trainer.trainOneDataBatch(batch_size, data_batch_gen)

# Copy the parameters from gen_training to dis_training and generator
copy_shared_parameters(gen_training_machine,
dis_training_machine)
copy_shared_parameters(gen_training_machine, generator_machine)
```

### 数据定义
这里数据没有通过dataprovider提供，而是在gan_trainer.py里面直接产生data_batch并以Arguments的形式提供给trainer。

```python
def prepare_generator_data_batch(batch_size, noise):
    label = numpy.ones(batch_size, dtype='int32')
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    inputs.setSlotIds(1, api.IVector.createVectorFromNumpy(label))
    return inputs

＃ Create data_batch for generator
data_batch_gen = prepare_generator_data_batch(batch_size, noise)
# Feed data_batch_gen into generator trainer
gen_trainer.trainOneDataBatch(batch_size, data_batch_gen)
```

### 算法配置

在这里，我们指定了模型的训练参数, 选择学习率和batch size。这里的beta1参数比默认值0.9小很多是为了使学习的过程更稳定。

```python
settings(
    batch_size=128,
    learning_rate=1e-4,
    learning_method=AdamOptimizer(beta1=0.5))

```

##训练模型
用MNIST手写数字图片训练对抗式生成网络可以用如下的命令：

```bash
$python gan_trainer.py -d mnist --useGpu 1
```

## 应用模型
图片由训练好的生成器生成。以下的代码将噪音z输入到生成器 G 当中，通过向前传递得到生成的图片。

```python
def get_fake_samples(generator_machine, batch_size, noise):
    gen_inputs = api.Arguments.createArguments(1)
    gen_inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    gen_outputs = api.Arguments.createArguments(0)
    generator_machine.forward(gen_inputs, gen_outputs, api.PASS_TEST)
    fake_samples = gen_outputs.getSlotValue(0).copyToNumpyMat()
    return fake_samples

# At the end of each pass, save the generated samples/images
fake_samples = get_fake_samples(generator_machine, batch_size, noise)
```

## 总结
本章中，我们介绍了对抗式生成网络的基本概念，训练方法以及如何用Paddle来实现。对抗式生成网络是现有生成模型当中非常重要的一种，它可以利用大量无标记数据来进行非监督学习，以寄希望能够得到对于复杂高维数据的一般有效的表示。


## 参考文献
1. Goodfellow I, Pouget-Abadie J, Mirza M, et al. [Generative adversarial nets](https://arxiv.org/pdf/1406.2661v1.pdf)[C] Advances in Neural Information Processing Systems. 2014
2. Radford A, Metz L, Chintala S. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434v2.pdf)[C] arXiv preprint arXiv:1511.06434. 2015
