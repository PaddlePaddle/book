# 对抗式生成网络

## 背景介绍
本章我们介绍对抗式生成网络，也称为Generative Adversarial Network(GAN) \[[1](#参考文献)\]。对抗式生成网络是生成模型 (generative model) 的一种，可以用非监督学习的办法来学习输入数据的分布，从而能达到产生和输入数据拥有同样概率分布的人造数据。这样的学习能力可以帮助机器完成图片自动生成、图像去噪、缺失图像补全和图像超分辨生成等工作。 

深度学习现有的方法大致可以分为两大类，判别模型（discriminative model）和生成模型（generative model）。

判别模型是在监督学习的条件下，把高维数据映射到一种低维空间表示（representation）里来进行分类（可参见前面几章的介绍），它直接对条件概率P(y|x)建模。像我们的前八章，都是判别模型。但用这种方法学到的表示一般只是对那一种目标任务有效果，而不能很好的转移到别的任务。同时监督学习的训练需要大量标记好的数据，很多时候不是很容易得到。

生成模型在监督学习和非监督学习的条件下都可以应用。在监督学习的条件下，生成模型是直接对联合概率P(X,Y)建模。在非监督学习的条件下，生成模型是对P(X)进行建模。生成模型背后的基本想法是，如果一个模型它能够生成和真实数据非常相近的数据，那么很可能它就学到了对于这种数据的一种很有效的表示。生成模型另一些实际用途包括，图像去噪，缺失图像补全，图像超分辨生成等等。在标记数据不够的时候，还可以用生成模型生成的数据来预训练模型。

生成模型一个重要的研究方向是图片生成。相比于生成文字，由于图片数据的维度更大并且数值是连续的，所以生成起来难度更大。关于图片生成的研究已经有比较久的历史，之前的方法有，受限玻尔兹曼机（Restricted Boltzmann Machine）\[[4](#参考文献)\]，深度玻尔兹曼机（Deep Boltzmann Machine）\[[5](#参考文献)\]，神经自回归分布估计（Neural Autoregressive Distribution Estimator）\[[6](#参考文献)\]等。但它们都无法生成看起来很真实的图片。

近年来由于深度学习的发展，出现了一些更有效的图片生成模型，一种是变分自编码器（variational autoencoder）\[[3](#参考文献)\]，它是在概率图模型（probabilistic graphical model）的框架下面搭建了一个生成模型，对数据有完整的概率描述，训练时是通过调节参数来最大化数据的概率。用这种方法产生的图片，虽然所对应的概率高，但很多时候看起来都比较模糊。另一种是像素循环神经网络（Pixel Recurrent Neural Network）\[[7](#参考文献)\]，它是通过根据周围的像素来一个像素一个像素的生成图片，但这种方法生成的图片在全局看来会不太一致。为了解决这些问题，人们又提出了本章所要介绍的另一种生成模型，对抗式生成网络。

在本章里，我们展对抗式生产网络的细节，以及如何用PaddlePaddle训练一个GAN模型。

## 效果展示
一个简单的例子是训练对抗式生成网络，使其学习产生MNIST手写数字的图片。由训练好的GAN模型产生的手写数字图片的例子画在图1中。

<p align="center">
    <img src="./mnist_sample.png" width="300" height="300"><br/>
    图1. GAN生成的MNIST例图
</p>

## 模型概览
对抗式生成网络的原理示意图在图2中画出，它由两部分组成：一个生成器（Generator）G 和一个分类器（Discriminator, 也称判别器）D，两者都是有多层神经网络构成的。生成器的输入是一个多维的已知概率分布的噪音 z(噪音的概率分布不取决于待生成样本，如可以服从正态分布)，通过神经网络变换，输出伪样本。分类器输的输入是真样本和伪样本，输出为分类结果为真样本和伪样本的概率。训练时生成器和分类器处于相互竞争对抗状态，生成器会尽量生成和真样本相近的伪样本让分类器无法分辨真伪，而分类器则会尽力去分辨伪样本。具体的损失函数如下：

$$\min_G\max_D \text{Loss} = \min_G\max_D \frac{1}{m}\sum_{i=1}^m[\log D(x^i) + log(1-D(G(z^i)))]$$

其中$x$是真实数据，$z$是已知概率分布的噪音。所以这个损失函数所代表的意义就是真实数据被分类为真的概率加上伪数据被分类为假的概率。分类器 D 目标是增加这个函数值，故公式里为max，而生成器 G 目标是减少这个函数值，故公式里为min。

<p align="center">
    <img src="./gan.png" width="500" height="300"><br/>
    图2. GAN模型原理示意图
    <a href="https://ishmaelbelghazi.github.io/ALI/">figure credit</a>
</p>

训练时，生成器和分类器会轮流通过随机梯度下降算法更新参数。生成器的目标函数是让自己产生的样本被分类器分类为真，而分类器的目标函数则是正确的区分真伪样本。当对抗式生成模型训练收敛到平衡态的时候，生成器会把输入的噪音分布转化成真的样本数据分布，而分类器则完全无法分辨真伪图片。

在最早的对抗式生成网络的论文中，生成器和分类器用的都是全联接层。在附带的代码gan_conf.py中，我们实现了一个类似的结构。生成器和分类器都是由三层全联接层构成，并且在某些全联接层后面加入了批标准化层（batch normalization）。所用网络结构在图3中给出。生成器的损失函数是其所生成的伪样本$x'$被判别器判定为真的概率，而判别器的损失函数是伪样本$x'$被判定为假的概率加上真样本$x$被判别为真的概率。

<p align="center">
    <img src="./gan_conf_graph.png" width="700" height="400"><br/>
    图3. GAN模型结构图
</p>

由于上面的这种网络都是由全联接层组成，所以没有办法很好的生成图片数据，也没有办法做的很深。所以在随后的论文中，人们提出了深度卷积对抗式生成网络（deep convolutional generative adversarial network or DCGAN）\[[2](#参考文献)\]。在DCGAN中，生成器 G 是由多个卷积转置层（transposed convolution）组成的，这样可以用更少的参数来生成质量更高的图片。具体网络结果可参见图4。而判别器是由多个卷积层组成。

<p align="center">
    <img src="./dcgan.png" width="700" height="300"><br/>
    图4. DCGAN生成器模型结构
    <a href="https://arxiv.org/pdf/1511.06434v2.pdf">figure credit</a>
</p>


## 数据准备
	
### 数据介绍与下载
这章会用到两种数据，一种是简单的人造数据，一种是图片。

人造数据是二维0到1之间的均匀分布，由下面的代码生成（numpy.random.rand会生成0-1均匀分布随机数）:

```python
# 合成2-D均匀分布数据 gan_trainer.py:114
def load_uniform_data():
    data = numpy.random.rand(1000000, 2).astype('float32')
    return data
```

图片数据是MNIST手写数字和CIFAR-10，可由下面的代码下载：

```bash
$cd data/
$./get_mnist_data.sh
$./download_cifar.sh
```

## 模型配置说明
由于对抗式生产网络涉及到多个神经网络，所以必须用paddle Python API来训练。下面的介绍也可以部分的拿来当作paddle Python API的使用说明。

### 数据定义
这里数据没有通过dataprovider提供，而是在gan_trainer.py里面直接产生data_batch并以Arguments的形式提供给trainer。

```python
def prepare_generator_data_batch(batch_size, noise):
	 # generator训练标签。根据前文的介绍，generator是为了让自己的生成的数据
	 # 被标记为真，所以这里的标签都统一生成1，也就是真
    label = numpy.ones(batch_size, dtype='int32')
    ＃ 数据是Arguments的类型，这里创建的一个有两个位置的Arguments
    inputs = api.Arguments.createArguments(2)
    ＃ 第一个Argument位置放noise
    inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    ＃ 第二个Argument位置放label
    inputs.setSlotIds(1, api.IVector.createVectorFromNumpy(label))
    return inputs

＃ 为generator训练创造数据
data_batch_gen = prepare_generator_data_batch(batch_size, noise)
# 把数据data_batch_gen传递给generator trainer
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
	
### 模型结构
在文件gan_conf.py当中我们定义了三个网络, **generator_training**, **discriminator_training** and **generator**. 和前文提到的模型结构的关系是：**discriminator_training** 是分类器，**generator** 是生成器，**generator_training** 是生成器加分类器因为训练生成器时需要用到分类器提供目标函数。这个对应关系在下面这段代码中定义：

```python
if is_generator_training:
    noise = data_layer(name="noise", size=noise_dim)
    # 函数generator定义了生成器的结构
    sample = generator(noise)

if is_discriminator_training:
    sample = data_layer(name="sample", size=sample_dim)

if is_generator_training or is_discriminator_training:
    label = data_layer(name="label", size=1)
    ＃ 函数discriminator定义了判别器的结构
    prob = discriminator(sample)
    cost = cross_entropy(input=prob, label=label)
    classification_error_evaluator(
        input=prob, label=label, name=mode + '_error')
    outputs(cost)

if is_generator:
    noise = data_layer(name="noise", size=noise_dim)
    outputs(generator(noise))
```

##训练模型
用MNIST手写数字图片训练对抗式生成网络可以用如下的命令：

```bash
$python gan_trainer.py -d mnist --use_gpu 1
```

训练中打印的日志信息如下：
```
d_pos_loss is 0.681067     d_neg_loss is 0.704936
d_loss is 0.693001151085    g_loss is 0.681496
...........d_pos_loss is 0.64475     d_neg_loss is 0.667874
d_loss is 0.656311988831    g_loss is 0.719081
...
I0105 17:15:48.346783 20517 TrainerInternal.cpp:165]  Batch=100 samples=12800 AvgCost=0.701575 CurrentCost=0.701575 Eval: generator_training_error=0.679219  CurrentEval: generator_training_error=0.679219 
.........d_pos_loss is 0.644203     d_neg_loss is 0.71601
d_loss is 0.680106401443    g_loss is 0.671118
....
I0105 17:16:37.172737 20517 TrainerInternal.cpp:165]  Batch=100 samples=12800 AvgCost=0.687359 CurrentCost=0.687359 Eval: discriminator_training_error=0.438359  CurrentEval: discriminator_training_error=0.438359 
```

其中d_pos_loss是判别器对于真实数据判别真的负对数概率，d_neg_loss是判别器对于伪数据判别为假的负对数概率，d_loss是这两者的平均值。g_loss是伪数据被判别器判别为真的负对数概率。对于对抗式生成网络来说，最好的训练情况是D和G的能力比较相近，也就是d_loss和g_loss在训练的前几个pass中数值比较接近（-log(0.5) = 0.693）。由于G和D是轮流训练，所以它们各自每过100个batch，都会打印各自的训练信息。

为了能够训练在gan_conf.py中定义的网络，我们需要如下几个步骤：初始化Paddle环境，解析设置，由设置创造GradientMachine以及由GradientMachine创造trainer。这几步分别由下面几段代码实现：

```python
import py_paddle.swig_paddle as api
# 初始化Paddle环境
api.initPaddle('--use_gpu=' + use_gpu, '--dot_period=10',
               '--log_period=100', '--gpu_id=' + args.gpu_id,
               '--save_dir=' + "./%s_params/" % data_source)

# 解析设置
gen_conf = parse_config(conf, "mode=generator_training,data=" + data_source)
dis_conf = parse_config(conf, "mode=discriminator_training,data=" + data_source)
generator_conf = parse_config(conf, "mode=generator,data=" + data_source)

# 由设置创造GradientMachine
dis_training_machine = api.GradientMachine.createFromConfigProto(
dis_conf.model_config)
gen_training_machine = api.GradientMachine.createFromConfigProto(
gen_conf.model_config)
generator_machine = api.GradientMachine.createFromConfigProto(
generator_conf.model_config)

# 由GradientMachine创造trainer
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
# 训练gen_training
gen_trainer.trainOneDataBatch(batch_size, data_batch_gen)

# 把gen_training中的参数同步到dis_training和generator当中
copy_shared_parameters(gen_training_machine,
dis_training_machine)
copy_shared_parameters(gen_training_machine, generator_machine)
```

## 应用模型
图片由训练好的生成器生成。以下的代码将噪音z输入到生成器 G 当中，通过向前传递得到生成的图片。

```python
# 噪音z是多维正态分布
def get_noise(batch_size, noise_dim):
    return numpy.random.normal(size=(batch_size, noise_dim)).astype('float32')

def get_fake_samples(generator_machine, batch_size, noise):
    gen_inputs = api.Arguments.createArguments(1)
    gen_inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    gen_outputs = api.Arguments.createArguments(0)
    generator_machine.forward(gen_inputs, gen_outputs, api.PASS_TEST)
    fake_samples = gen_outputs.getSlotValue(0).copyToNumpyMat()
    return fake_samples

# 在每个pass的最后，保存生成的图片
noise = get_noise(batch_size, noise_dim)
fake_samples = get_fake_samples(generator_machine, batch_size, noise)
```

## 总结
本章中，我们介绍了对抗式生成网络的基本概念，训练方法以及如何用Paddle来实现。对抗式生成网络是现有生成模型当中非常重要的一种，它可以利用大量无标记数据来进行非监督学习，以寄希望能够得到对于复杂高维数据的一般有效的表示。


## 参考文献
1. Goodfellow I, Pouget-Abadie J, Mirza M, et al. [Generative adversarial nets](https://arxiv.org/pdf/1406.2661v1.pdf)[C] Advances in Neural Information Processing Systems. 2014
2. Radford A, Metz L, Chintala S. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434v2.pdf)[C] arXiv preprint arXiv:1511.06434. 2015
3. Kingma D.P. and Welling M. [Auto-encoding variational bayes](https://arxiv.org/pdf/1312.6114v10.pdf)[C] arXiv preprint arXiv:1312.6114. 2013
4. Hinton G and Salakhutdinov R. [Reducing the dimensionality of data with neural networks](https://www.cs.toronto.edu/~hinton/science.pdf) Science 313.5786. 2006
5. Salakhutdinov R and Hinton G. [Deep Boltzmann Machines](http://www.jmlr.org/proceedings/papers/v5/salakhutdinov09a/salakhutdinov09a.pdf)[J] AISTATS. Vol. 1. 2009
6. Larochelle H and Murray I. [The Neural Autoregressive Distribution Estimator](http://www.jmlr.org/proceedings/papers/v15/larochelle11a/larochelle11a.pdf) AISTATS. Vol. 1. 2011.
7. van den Oord A, Kalchbrenner N and Kavukcuoglu K. [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759v3.pdf) arXiv preprint arXiv:1601.06759 (2016).
