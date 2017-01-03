# 对抗式生成网络

## 背景介绍
本章我们介绍对抗式生成网络，也称为Generative Adversarial Network(GAN) \[[1](#参考文献)\]。对抗式生成网络是生成模型 (generative model) 的一种，可以用非监督学习的办法来学习输入数据的分布，从而能达到产生和输入数据拥有同样概率分布的人造数据。这样的学习能力可以帮助机器完成图片自动生成、图像去噪、缺失图像补全和图像超分辨生成等工作。

现在大部分利用深度学习成功的例子都是在监督学习的条件下，把高维数据映射到一种低维空间表示（representation）里来进行分类（可参见前面几章的介绍）。这种方法也叫判别模型（discriminative model），它直接对条件概率P(y|x)建模。像我们的前八章，都是判别模型。但用这种方法学到的表示一般只是对那一种目标任务有效果，而不能很好的转移到别的任务。同时监督学习的训练需要大量标记好的数据，很多时候不是很容易得到。

生成模型背后的基本想法是，如果一个模型它能够生成和真实数据非常相近的数据，那么很可能它就学到了对于这种数据的一种很有效的表示。生成模型另一些实际用途包括，图像去噪，缺失图像补全，图像超分辨生成等等。在标记数据不够的时候，还可以用生成模型生成的数据来预训练模型。

近年来有一些有趣的图片生成模型，一种是变分自编码器（variational autoencoder）\[[3](#参考文献)\]，它是在概率图模型（probabilistic graphical model）的框架下面搭建了一个生成模型，对数据有完整的概率描述（即对P(x)进行建模），训练时是通过调节参数来最大化数据的概率。用这种方法产生的图片，虽然所对应的概率高，但很多时候看起来都比较模糊。为了解决这个问题，人们又提出了本章所要介绍的另一种生成模型，对抗式生成网络。

在本章里，我们展对抗式生产网络的细节，以及如何用PaddlePaddle训练一个GAN模型。

## 效果展示
一个简单的例子是训练对抗式生成网络，使其学习产生MNIST手写数字的图片。由训练好的GAN模型产生的手写数字图片的例子画在图1中。

<p align="center">
    <img src="./mnist_sample.png" width="300" height="300"><br/>
    图1. GAN生成的MNIST例图
</p>

## 模型概览
对抗式生成网络的大致结构在图2中画出，它由两部分组成：一个生成器（Generator）G 和一个分类器（Discriminator, 也称判别器）D，两者都是有多层神经网络构成的。生成器的输入是一个多维的已知概率分布的噪音 z(噪音的概率分布不取决于待生成样本，如可以服从正态分布)，通过神经网络变换，输出伪样本。分类器输的输入是真样本和伪样本，输出为分类结果为真样本和伪样本的概率。训练时生成器和分类器处于相互竞争对抗状态，生成器会尽量生成和真样本相近的伪样本让分类器无法分辨真伪，而分类器则会尽力去分辨伪样本。具体的损失函数如下：

$$\min_G\max_D \text{Loss} = \min_G\max_D \frac{1}{m}\sum_{i=1}^m[\log D(x^i) + log(1-D(G(z^i)))]$$

其中$x$是真实数据，$z$是已知概率分布的噪音。所以这个损失函数所代表的意义就是真实数据被分类为真的概率加上伪数据被分类为假的概率。分类器 D 目标是增加这个函数值，故公式里为max，而生成器 G 目标是减少这个函数值，故公式里为min。

<p align="center">
    <img src="./gan.png" width="500" height="300"><br/>
    图2. GAN模型结构
    <a href="https://ishmaelbelghazi.github.io/ALI/">figure credit</a>
</p>

训练时，生成器和分类器会轮流通过随机梯度下降算法更新参数。生成器的目标函数是让自己产生的样本被分类器分类为真，而分类器的目标函数则是正确的区分真伪样本。当对抗式生成模型训练收敛到平衡态的时候，生成器会把输入的噪音分布转化成真的样本数据分布，而分类器则完全无法分辨真伪图片。

在最早的对抗式生成网络的论文中，生成器和分类器用的都是全联接层，所以没有办法很好的生成图片数据，也没有办法做的很深。所以在随后的论文中，人们提出了深度卷积对抗式生成网络（deep convolutional generative adversarial network or DCGAN）\[[2](#参考文献)\]。在DCGAN中，生成器 G 是由多个卷积转置层（transposed convolution）组成的，这样可以用更少的参数来生成质量更高的图片。具体网络结果可参见图3。

<p align="center">
    <img src="./dcgan.png" width="700" height="300"><br/>
    图3. DCGAN生成器模型结构
    <a href="https://arxiv.org/pdf/1511.06434v2.pdf">figure credit</a>
</p>


## 数据准备
	
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
3. Kingma D.P. and Welling M. [Auto-encoding variational bayes](https://arxiv.org/pdf/1312.6114v10.pdf)[C] arXiv preprint arXiv:1312.6114. 2013