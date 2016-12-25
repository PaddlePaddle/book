# 语义角色标注（Semantic Role Labeling）

## 背景介绍

语义角色标注任务就是以句子的谓词为中心，研究句子中各成分与谓词之间的关系，并且用语义角色来描述他们之间的关系，这是一种以句子为单位浅层语义分析技术，也就是，不对句子所包含的语义信息进行深入地分析，而只是分析句子的谓词-论元结构。

请看下面的例子：

$$\mbox{[小明]}_{\mbox{Agent}}\mbox{[昨天]}_{\mbox{Time}}\mbox{在[公园]}_{\mbox{Location}}\mbox{[遇到]}_{\mbox{Predicate}}\mbox{了[小红]}_{\mbox{Patient}}\mbox{。}$$

在上面的句子中，“遇到” 是谓词（Predicate，通常简写为“Pred”），代表了一个事件的核心，“小明”是施事者（Agent），“小红”是受事者（Patient），“昨天” 是事件发生的时间（Time），“公园”是时间发生的地点（Location）。

通过这个例子可以看出，语义角色标注就是要分析出句子描述的事件：时间的参与者（包括施事者、受事者）、事件发生的时间、地点和原因等。

在经典方法中，语义角色标注通常又包括三个流程，一是句子中谓词的识别（一般为动词）；二是谓词语义的判定（如“打人”、“打饭”中的“打”具有不同的语义），三是谓词支配词识别及角色判定（施事者 Agent、受事者 Patient、方式 Manner、地点 Location、时间 Time 等）。这些流程大多建立在句法分析的基础之上。然而，目前技术下的句法分析准确率不高，句法分析的细微错误也会导致语义角色标注的错误，造成语义角色标注的准确率也收到很大地限制，这也是语义角色标注任务面临的最主要挑战。

为了回避无法获得准确率较高的结构树或依存结构树所造成的困难，研究提出了基于语块的语义角色标注方法。也是我们这篇文章所要介绍的方法。基于语块的语义角色标注方法将语义标注方法作为一个序列标注问题来解决，是一个相对简单的过程。一般采用IBO表示方式来定义序列标注的标签集，将不同的语块赋予不同的标签。即：对于一个角色为A的论元，将它所包含的第一个语块赋予标签B-A，将它所包含的其它语块赋予标签I-A，不属于任何论元的语块赋予标签O。根据序列标注的结果就可以直接得到语义角色标注的结果，而且论元识别和论元标注通常作为一个过程同时实现。

# 模型概览

在这篇文章里，语义角色标注任务被规范化地描述成为一个序列标注问题，在开始构建我们的序列标注模型之前，我们首先介绍三个重要模块。

## 双向循环神经网络模型（Bidirectional Recurrent Neural Network）
循环神经网络（Recurrent Neural Network）是一种对序列建模的重要模型，在自然语言处理任务中有着广泛地应用。不同于传统的前馈神经网络(Feed-forward Neural Networks)，RNN 引入了循环，能够处理输入之间前后关联的问题。例如，在语言中，由于句子前后单词并不是独立存在，我们要标记句子中的下一个词的语义角色，通常都依赖句子前面的词。

RNN 之所以称为循环神经网路，是因为在 RNN 模型中，隐藏层之间的是有连接的，即：隐藏层的输入不仅包括当前时刻输入层的输出，还包括上一时刻隐藏层的输出，网络会对前面时刻的信息进行记忆，并用于当前时刻输出的计算中。于是，当前时刻输出也收到前面时刻的输出的影响。

理论上，RNN $t$ 时刻的输出编码了到$t$ 时刻之前所有历史信息，然而，标准的神经网络通常都忽略了未来时刻的上下文信息的引入。如果能像访问历史上下文信息一样，同样访问未来上下文信息，对于许多序列学习任务都是非常有益的。双向 RNN 就是解决这一问题的一种简单有效的方法，由 Bengio 等人在论文\[[1](#参考文献),[2](#参考文献)\]中提出。

双向 RNN 每一个训练序列向前和向后分别是两个循环神经网络（RNN），而且这两个都连接着一个输出层。这个结构提供给输出层输入序列中每一个点的完整的过去和未来的上下文信息。下图展示的是一个沿着时间展开的双向循环神经网络。双向 RNN 包含一个前向（forward）和一个后向（backward）RNN 单元，其中包含六个权值：输入到前向隐层和后向隐层（$w_1, w_3$），隐层到隐层自己（$w_2,w_5$），前向隐层和后向隐层到输出层（$w_4, w_6$），在这里，前向隐层和后向隐层之间没有信息流。

<div  align="center">    
<img src="./image/bi-rnn.png" width = "35%" height = "35%" align=center />
<center> 图1. 按时间步展开的双向RNN网络 </center>
</div>



理论上，RNN 能够处理任意长度的输入序列，但是在实践中，由于梯度消失和梯度发散问题的存在，建模超长的输入序列依然存在许多优化上的难题。为了能够更好地建模长序列中存在的长程依赖关系，研究者们设计了带有门机制，更加精巧的 LSTM（Long Short Term Memory）\[[3](#参考文献)\] 和 GRU (Gated Recurrent Unit) \[[2](#参考文献)\]，于是我们可以利用双向 RNN 的思想，同样道理构建双向 LSTM 和双向 GRU。

关于这些不同 RNN 单元的比较，可以参考论文 \[[4](#参考文献)\] 。

## Stacked Recurrent Neural Network

深度网络能够帮助我们学习层次化特征，网络的上层在下层已经学习到的初级特征基础上，学习更复杂的高级特征。堆叠多个 RNN 单元（可以是：Simple RNN ， LSTM 或者 GRU）同样能够带来这样的好处。

RNN 等价于一个展开地前向网络，于是，通常人们会认为 RNN 在时间轴上是一个真正的“深层网络”。然而，在循环神经网络中，我们对网络层数的定义并非如此直接。

输入特征经过一次非线性映射，我们称之为神经网络的一层。按照这样的约定，可以看到，尽管 RNN 沿时间轴展开后等价于一个非常“深”的前馈网络，但由于 RNN 各个时间步参数共享，$t-1$ 时刻隐藏层输出到 $t$ 时刻的映射，始终只经过了一次非线性映射，也就是说 ：RNN 对状态转移的建模是 “浅” 的。纵向堆叠多个 RNN 单元 ，令前一个 RNN $t$ 时刻的输出，成为下一个 RNN 单元 $t$ 时刻的输入，帮助我们构建起一个深层的 RNN 网络。和单层 RNN 网络相比，深层 RNN 网络能够更好地建模跨不同时间步的特征\[[5](#参考文献)\]。

## 条件随机场 (Conditional Random Field)

条件随机场 （Conditional Random Filed， CRF）是一种用来标注和划分序列结构数据的概率化结构模型，可以看作是一个无向图模型，或者马尔科夫随机场。我们首先来看看一般的条件随机场是如何定义的。

**条件随机场** : 设 $G = (V, E)$ 是一个无向图， $V$ 是结点的集合，$E$ 是无向边的集合。$V$ 中的每个结点对应一个随机变量 $Y_{v}$， $Y = \{Y_{v} | v \in V\}$，其取值范围为可能的标记集合 $\{y\}$，如果以观测序列 $X$ 为条件，每个随机变量 $Y_{v}$ 都满足以下马尔科夫特性：
$$p(Y_{v}|X, Y_{\omega}, \omega \not= v) = p(Y_{v} | X, Y_{\omega} , \omega \sim v)$$
其中，$\omega \sim v$ 表示两个结点在图 $G$ 中是临近结点，那么，$(X, Y)$ 是一个条件随机场。


CRF 用于序列标注由 Lafferty 等人 \[[7](#参考文献)\] 于2001年提出。 理论上，只要在标记序列中表示了一定的条件独立性， $G$ 的图结构可以是任意的。在序列标注任务中，只考虑 $X$ 和 $Y$ 具有相的图结构：都是一个序列，于是建模一个序列可以形成如图2所示的一个简单的链式结构图。因此，序列标注问题使用的是一种定义在线性链上的特殊条件随机场，称之为线性链条件随机场（linear chain conditional random field）。

<div  align="center">    
<img src="./image/linear_chain_crf.png" width = "35%" height = "35%" align=center />
<center>  图2. $X$ 和 $Y$ 具有相同结构的线性链条件随机场 </center> 
</div>

**线性链条件随机场** ：设 $X = (X_{1}, X_{2}, ... , X_{n})$，$Y = {Y_{1}, Y_{2}, ... , Y_{n}}$ 均为线性链表示的随机变量序列，若在给定随机变量序列 $X$ 的条件下，随机变量序列 $Y$ 的条件概率分布 $P(Y|X)$ 满足马尔科夫性：$$p(Y_{i}|X, Y_{1}, ... , Y_{i - 1}, Y_{i + 1}, ... , Y_{n}) = p(Y_{i} | X, Y_{i - 1}, Y_{i + 1})$$
$$i = 1, 2, ..., n \mbox{（在} i = 1 \mbox{和} n \mbox{时只考虑单边）}$$，则称$P(Y|X)$为线性链条件随机场。$X$表示输入的观测序列，$Y$ 表示对应的输出标记序列。

根据线性链条件随机场上的因子分解定理 \[[7](#参考文献)\]，在给定观测序列 $X$ 时，一个特定标记序列 $Y$ 的概率可以定义为：
$$\frac{1}{Z(x)} exp(\sum_{j} \lambda_{j}t_{j} (y_{i - 1}, y_{i}, X, i) + \sum_{k} (\mu_k s_k (y_i, X, i)))$$ $$Z(x) = \sum_y exp(\sum_{i, k} \lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{i, l}\mu_l s_l (y_i, x, i))$$ 

式中$Z(x)$是规范化因子。$t_j$ 是定义在边上的特征函数，依赖于当前和前一个位置，称为转移特征。$s_k$ 是定义在结点上的特征函数，称为状态特征，依赖于当前位置。$\lambda_j$ 和 $\mu_k$ 是对应的权值。于是，条件随机场完全由特征函数 $t_j$ ，$s_k$ 和它们对应的权值$\lambda_j$ ，$\mu_l$ 确定。

学习时，对于给定的输入输出变量序列 $X$（表示需要标注的观测序列）和输出变量序列 $Y$ （表示标记序列），利用训练数据集，通过极大似然估计或是正则化的极大似然估计可以求解得到条件概率模型 $\bar{P}(Y|X)$；测试时，对于给定的输入序列 $x$，求出条件概率$\bar{P}(Y|X)$最大的输出序列 $\bar{y}$。关于 CRF 的训练和解码更多的解释，可以参考\[[8](#参考文献)\]。

# 数据准备
## 数据介绍与下载
在此教程中我们选用 [CoNLL 2005](http://www.cs.upc.edu/~srlconll/) 公共任务中的语义角色标注数据集\[[9](#参考文献)\]。

## 提供数据给 PaddlePaddle

```
@provider(
    init_hook=hook,
    should_shuffle=True,
    calc_batch_size=get_batch_size,
    can_over_batch_size=True,
    cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            sentence, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2,  mark, label = \
                line.strip().split('\t')

            words = sentence.split()
            sen_len = len(words)
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in words]

            predicate_slot = [settings.predicate_dict.get(predicate)] * sen_len
            ctx_n2_slot = [settings.word_dict.get(ctx_n2, UNK_IDX)] * sen_len
            ctx_n1_slot = [settings.word_dict.get(ctx_n1, UNK_IDX)] * sen_len
            ctx_0_slot = [settings.word_dict.get(ctx_0, UNK_IDX)] * sen_len
            ctx_p1_slot = [settings.word_dict.get(ctx_p1, UNK_IDX)] * sen_len
            ctx_p2_slot = [settings.word_dict.get(ctx_p2, UNK_IDX)] * sen_len

            marks = mark.split()
            mark_slot = [int(w) for w in marks]

            label_list = label.split()
            label_slot = [settings.label_dict.get(w) for w in label_list]
            yield word_slot, ctx_n2_slot, ctx_n1_slot, \
                  ctx_0_slot, ctx_p1_slot, ctx_p2_slot, predicate_slot, mark_slot, label_slot


```
## 模型配置说明
## 数据定义

在模型配置中，首先定义通过 define_py_data_sources2 从 dataprovider 中读入数据。


```
define_py_data_sources2(
        train_list=train_list_file,
        test_list=test_list_file,
        module='dataprovider',
        obj='process',
        args={
            'word_dict': word_dict,
            'label_dict': label_dict,
            'predicate_dict': predicate_dict}
)
```
## 算法配置
在这里，我们指定了模型的训练参数, 选择L2正则项稀疏、学习率和batch size。

```
settings(
    batch_size=150,
    learning_method=MomentumOptimizer(momentum=0),
    learning_rate=2e-2,
    regularization=L2Regularization(8e-4),
    is_async=False,
    model_average=ModelAverage(average_window=0.5, max_average_window=10000)
)
```

## 模型结构
# 训练模型
执行sh train.sh进行模型的训练。其中指定了总共需要执行500个pass。

```
paddle train \
  --config=./db_lstm.py \
  --save_dir=./output \
  --trainer_count=4 \
  --dot_period=500 \
  --log_period=10 \
  --num_passes=500 \
  --use_gpu=false \
  --show_parameter_stats_period=10 \
  --test_all_data_in_one_period=1 \
2>&1 | tee 'train.log'
```
一轮训练 log 示例如下所示，经过 150 个pass，得到平均 error 为 0.0516055。

```
I1224 18:11:53.661479  1433 TrainerInternal.cpp:165]  Batch=880 samples=145305 AvgCost=2.11541 CurrentCost=1.8645 Eval: __sum_evaluator_0__=0.607942  CurrentEval: __sum_evaluator_0__=0.59322
I1224 18:11:55.254021  1433 TrainerInternal.cpp:165]  Batch=885 samples=146134 AvgCost=2.11408 CurrentCost=1.88156 Eval: __sum_evaluator_0__=0.607299  CurrentEval: __sum_evaluator_0__=0.494572
I1224 18:11:56.867604  1433 TrainerInternal.cpp:165]  Batch=890 samples=146987 AvgCost=2.11277 CurrentCost=1.88839 Eval: __sum_evaluator_0__=0.607203  CurrentEval: __sum_evaluator_0__=0.590856
I1224 18:11:58.424069  1433 TrainerInternal.cpp:165]  Batch=895 samples=147793 AvgCost=2.11129 CurrentCost=1.84247 Eval: __sum_evaluator_0__=0.607099  CurrentEval: __sum_evaluator_0__=0.588089
I1224 18:12:00.006893  1433 TrainerInternal.cpp:165]  Batch=900 samples=148611 AvgCost=2.11148 CurrentCost=2.14526 Eval: __sum_evaluator_0__=0.607882  CurrentEval: __sum_evaluator_0__=0.749389
I1224 18:12:00.164089  1433 TrainerInternal.cpp:181]  Pass=0 Batch=901 samples=148647 AvgCost=2.11195 Eval: __sum_evaluator_0__=0.60793
```

# 总结
# 参考文献
1. Cho K, Van Merriënboer B, Gulcehre C, et al. [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)[J]. arXiv preprint arXiv:1406.1078, 2014.
2. Bahdanau D, Cho K, Bengio Y. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)[J]. arXiv preprint arXiv:1409.0473, 2014.
3. Hochreiter S, Schmidhuber J. [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)[J]. Neural computation, 1997, 9(8): 1735-1780.
4. Chung J, Gulcehre C, Cho K H, et al. [Empirical evaluation of gated recurrent neural networks on sequence modeling](https://arxiv.org/abs/1412.3555)[J]. arXiv preprint arXiv:1412.3555, 2014.
5. Pascanu R, Gulcehre C, Cho K, et al. [How to construct deep recurrent neural networks](https://arxiv.org/abs/1312.6026)[J]. arXiv preprint arXiv:1312.6026, 2013.
6. Sutskever I, Vinyals O, Le Q V. [Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf)[C]//Advances in neural information processing systems. 2014: 3104-3112.
7. Lafferty J, McCallum A, Pereira F. [Conditional random fields: Probabilistic models for segmenting and labeling sequence data](http://www.jmlr.org/papers/volume15/doppa14a/source/biblio.bib.old)[C]//Proceedings of the eighteenth international conference on machine learning, ICML. 2001, 1: 282-289.
8. 李航. 统计学习方法[J]. 清华大学出版社, 北京, 2012.
9. Palmer M, Gildea D, Kingsbury P. [The proposition bank: An annotated corpus of semantic roles](https://www.cs.rochester.edu/~gildea/palmer-propbank-cl.pdf)[J]. Computational linguistics, 2005, 31(1): 71-106.