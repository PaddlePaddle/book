# Word2Vec

This is intended as a reference tutorial. The source code of this tutorial lives on [book/word2vec](https://github.com/PaddlePaddle/book/tree/develop/word2vec).

For instructions on getting started with PaddlePaddle, see [PaddlePaddle installation guide](http://www.paddlepaddle.org/doc_cn/build_and_install/index.html).

## Background Introduction

This section introduces the concept of **word embedding**, which is a vector representation of words. It is a popular technique used in natural language processing. Word embeddings support many Internet services, including search engines, advertising systems, and recommendation systems.

### One-Hot Vectors

Building these services requires us to quantify the similarity between two words or paragraphs. This calls for a new representation of all the words to make them more suitable for computation. An obvious way to achieve this is through the vector space model, where every word is represented as an **one-hot vector**.

For each word, its vector representation has the corresponding entry in the vector as 1, and all other entries as 0. The lengths of one-hot vectors match the size of the dictionary. Each entry of a vector corresponds to the presence (or absence) of a word in the dictionary.

One-hot vectors are intuitive, yet they have limited usefulness. Take the example of an Internet advertising system: Suppose a customer enters the query "Mother's Day", while an ad bids for the keyword carnations". Because the one-hot vectors of these two words are perpendicular, the metric distance (either Euclidean or cosine similarity) between them would indicate  little relevance. However, *we* know that these two queries are connected semantically, since people often gift their mothers bundles of carnation flowers on Mother's Day. This discrepancy is due to the low information capacity in each vector. That is, comparing the vector representations of two words does not assess their relevance sufficiently. To calculate their similarity accurately, we need more information, which could be learned from large amounts of data through machine learning methods.

Like many machine learning models, word embeddings can represent knowledge in various ways. Another model may project an one-hot vector to an embedding vector of lower dimension e.g. $embedding(mother's day) = [0.3, 4.2, -1.5, ...], embedding(carnations) = [0.2, 5.6, -2.3, ...]$. Mapping one-hot vectors onto an embedded vector space has the potential to bring the embedding vectors of similar words (either semantically or usage-wise) closer to each other, so that the cosine similarity between the corresponding vectors for words like "Mother's Day" and "carnations" are no longer zero.

A word embedding model could be a probabilistic model, a co-occurrence matrix model, or a neural network. Before people started using neural networks to generate word embedding, the traditional method was to calculate a co-occurrence matrix $X$ of words. Here, $X$ is a $|V| \times |V|$ matrix, where $X_{ij}$ represents the co-occurrence times of the $i$th and $j$th words in the vocabulary `V` within all corpus, and $|V|$ is the size of the vocabulary. By performing matrix decomposition on $X$ e.g. Singular Value Decomposition \[[5](#References)\]

$$X = USV^T$$

the resulting $U$ can be seen as the word embedding of all the words.

However, this method suffers from many drawbacks:
1) Since many pairs of words don't co-occur, the co-occurrence matrix is sparse. To achieve good performance of matrix factorization, further treatment on word frequency is needed;
2) The matrix is large, frequently on the order of $10^6*10^6$;
3) We need to manually filter out stop words (like "although", "a", ...), otherwise these frequent words will affect the performance of matrix factorization.

The neural network based model does not require storing huge hash tables of statistics on all of the corpus. It obtains the word embedding by learning from semantic information, hence could avoid the aforementioned problems in the traditional method. In this chapter, we will introduce the details of neural network word embedding model and how to train such model in PaddlePaddle.

## Results Demonstration

In this section, after training the word embedding model, we could use the data visualization algorithm $t-$SNE\[[4](#reference)\] to draw the word embedding vectors after projecting them onto a two-dimensional space (see figure below). From the figure we could see that the semantically relevant words -- *a*, *the*, and *these* or *big* and *huge* -- are close to each other in the projected space, while irrelevant words -- *say* and *business* or *decision* and *japan* -- are far from each other.

<p align="center">
	<img src = "image/2d_similarity.png" width=400><br/>
	Figure 1. Two dimension projection of word embeddings
</p>

### Cosine Similarity

On the other hand, we know that the cosine similarity between two vectors falls between $[-1,1]$. Specifically, the cosine similarity is 1 when the vectors are identical, 0 when the vectors are perpendicular, -1 when the are of opposite directions. That is, the cosine similarity between two vectors scales with their relevance. So we can calculate the cosine similarity of two word embedding vectors to represent their relevance:

```
please input two words: big huge
similarity: 0.899180685161

please input two words: from company
similarity: -0.0997506977351
```

The above results could be obtained by running `calculate_dis.py`, which loads the words in the dictionary and their corresponding trained word embeddings. For detailed instruction, see section [Model Application](#Model Application).


## Model Overview

In this section, we will introduce three word embedding models: N-gram model, CBOW, and Skip-gram, which all output the frequency of each word given its immediate context.

For N-gram model, we will first introduce the concept of language model, and implement it using PaddlePaddle in section [Model Training](#Model Training).

The latter two models, which became popular recently, are neural word embedding model developed by Tomas Mikolov at Google \[[3](#reference)\]. Despite their apparent simplicity, these models train very well.

### Language Model

Before diving into word embedding models, we will first introduce the concept of **language model**. Language models build the joint probability function $P(w_1, ..., w_T)$ of a sentence, where $w_i$ is the i-th word in the sentence. The goal is to give higher probabilities to meaningful sentences, and lower probabilities to meaningless constructions.

In general, models that generate the probability of a sequence can be applied to many fields, like machine translation, speech recognition, information retrieval, part-of-speech tagging, and handwriting recognition. Take information retrieval, for example. If you were to search for "how long is a football bame" (where bame is a medical noun), the search engine would have asked if you had meant "how long is a football game" instead. This is because the probability of "how long is a football bame" is very low according to the language model; in addition, among all of the words easily confused with "bame", "game" would build the most probable sentence.

#### Target Probability
For language model's target probability $P(w_1, ..., w_T)$, if the words in the sentence were to be independent, the joint probability of the whole sentence would be the product of each word's probability: 

$$P(w_1, ..., w_T) = \prod_{t=1}^TP(w_t)$$

However, the frequency of words in a sentence typically relates to the words before them, so canonical language models are constructed using conditional probability in its target probability:

$$P(w_1, ..., w_T) = \prod_{t=1}^TP(w_t | w_1, ... , w_{t-1})$$ 


### N-gram neural model 

In computational linguistics, n-gram is an important method to represent text. An n-gram represents a contiguous sequence of n consecutive items given a text. Based on the desired application scenario, each item could be a letter, a syllable or a word. The N-gram model is also an important method in statistical language modeling. When training language models with n-grams, the first (n-1) words of an n-gram are used to predict the *n*th word.

Yoshua Bengio and other scientists describe how to train a word embedding model using neural network in the famous paper of Neural Probabilistic Language Models \[[1](#reference)\] published in 2003. The Neural Network Language Model (NNLM) described in the paper learns the language model and word embedding simultaneously through a linear transformation and a non-linear hidden connection. That is, after training on large amounts of corpus, the model learns the word embedding; then, it computes the probability of the whole sentence, using the embedding. This type of language model can overcome the **curse of dimensionality** i.e. model inaccuracy caused by the difference in dimensionality between training and testing data. Note that the term *neural network language model* is ill-defined, so we will not use the name NNLM but only refer to it as *N-gram neural model* in this section.

We have previously described language model using conditional probability, where the probability of the *t*-th word in a sentence depends on all $t-1$ words before it. Furthermore, since words further prior have less impact on a word, and every word within an n-gram is only effected by its previous n-1 words, we have:

$$P(w_1, ..., w_T) = \prod_{t=n}^TP(w_t|w_{t-1}, w_{t-2}, ..., w_{t-n+1})$$

Given some real corpus in which all sentences are meaningful, the n-gram model should maximize the following objective function: 

$$\frac{1}{T}\sum_t f(w_t, w_{t-1}, ..., w_{t-n+1};\theta) + R(\theta)$$

where $f(w_t, w_{t-1}, ..., w_{t-n+1})$ represents the conditional probability of the current word $w_t$ given its previous $n-1$ words, and $R(\theta)$ represents parameter regularization term. 

<p align="center">	
   	<img src="image/nnlm.png" width=500><br/>
   	Figure 2. N-gram neural network model
</p>

Figure 2 shows the N-gram neural network model. From the bottom up, the model has the following components:

 - For each sample, the model gets input $w_{t-n+1},...w_{t-1}$, and outputs the probability that the t-th word is one of `|V|` in the dictionary.
 
 Every input word $w_{t-n+1},...w_{t-1}$ first gets transformed into word embedding $C(w_{t-n+1}),...C(w_{t-1})$ through a transformation matrix. 
 
 - All the word embeddings concatenate into a single vector, which is mapped (nonlinearly) into the $t$-th word hidden representation:

	$$g=Utanh(\theta^Tx + b_1) + Wx + b_2$$
	
   where $x$ is the large vector concatenated from all the word embeddings representing the context; $\theta$, $U$, $b_1$, $b_2$ and $W$ are parameters connecting word embedding layers to the hidden layers. $g$ represents the unnormalized probability of the output word, $g_i$ represents the unnormalized probability of the output word being the i-th word in the dictionary. 

 - Based on the definition of softmax, using normalized $g_i$, the probability that the output word is $w_t$ is represented as:
 
  $$P(w_t | w_1, ..., w_{t-n+1}) = \frac{e^{g_{w_t}}}{\sum_i^{|V|} e^{g_i}}$$
 
 - The cost of the entire network is a multi-class cross-entropy and can be described by the following loss function

   $$J(\theta) = -\sum_{i=1}^N\sum_{c=1}^{|V|}y_k^{i}log(softmax(g_k^i))$$ 

   where $y_k^i$ represents the true label for the $k$-th class in the $i$-th sample ($0$ or $1$), $softmax(g_k^i)$ represents the softmax probability for the $k$-th class in the $i$-th sample.

### Continuous Bag-of-Words model(CBOW) 

CBOW model predicts the current word based on the N words both before and after it. When $N=2$, the model is as the figure below:

<p align="center">	
	<img src="image/cbow.png" width=250><br/>
	Figure 3. CBOW model
</p>

Specifically, by ignoring the order of words in the sequence, CBOW uses the average value of the word embedding of the context to predict the current word:

$$\text{context} = \frac{x_{t-1} + x_{t-2} + x_{t+1} + x_{t+2}}{4}$$

where $x_t$ is the word embedding of the t-th word, classification score vector is $z=U*\text{context}$, the final classification $y$ uses softmax and the loss function uses multi-class cross-entropy.

### Skip-gram model 

The advantages of CBOW is that it smooths over the word embeddings of the context and reduces noise, so it is very effective on small dataset. Skip-gram uses a word to predict its context and get multiple context for the given word, so it can be used in larger datasets. 

<p align="center">	
	<img src="image/skipgram.png" width=250><br/>
	Figure 4. Skip-gram model
</p>


As illustrated in the figure above, skip-gram model maps the word embedding of the given word onto $2n$ word embeddings (including $n$ words before and $n$ words after the given word), and then combine the classification loss of all those $2n$ words by softmax. 

## Data Preparation

## Model Configuration
	
## Model Training

## Model Application
 
## Conclusion

This chapter introduces word embedding, the relationship between language model and word embedding, and how to train neural networks to learn word embedding.

In information retrieval, the relevance between the query and document keyword can be computed through the cosine similarity of their word embeddings. In grammar analysis and semantic analysis, a previously trained word embedding can initialize models for better performance. In document classification, clustering the word embedding can group synonyms in the documents. We hope that readers can use word embedding models in their work after reading this chapter.


## Referenes
1. Bengio Y, Ducharme R, Vincent P, et al. [A neural probabilistic language model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)[J]. journal of machine learning research, 2003, 3(Feb): 1137-1155.
2. Mikolov T, Kombrink S, Deoras A, et al. [Rnnlm-recurrent neural network language modeling toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-demo.pdf)[C]//Proc. of the 2011 ASRU Workshop. 2011: 196-201.
3. Mikolov T, Chen K, Corrado G, et al. [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf)[J]. arXiv preprint arXiv:1301.3781, 2013.
4. Maaten L, Hinton G. [Visualizing data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)[J]. Journal of Machine Learning Research, 2008, 9(Nov): 2579-2605.
5. https://en.wikipedia.org/wiki/Singular_value_decomposition

<br/>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">本教程</span> 由 <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a> 创作，采用 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享 署名-非商业性使用-相同方式共享 4.0 国际 许可协议</a>进行许可。
