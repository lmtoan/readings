# [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Thesis
---
This paper builds upon the literature of Neural Machine Translation (NMT) which has emerged as the most promising machine translation approach in recent years. Neural methods display superior performance on adequacy and fluency benchmarks of various language pairs. NMT research accelerates with the introduction of sequence-to-sequence model variations that leverage the Recurrent Neural Network (RNN) architecture. To improve translation quality for longer sentences, attention mechanism is one of several refinements that help NMT systems frequently yield state-of-the-art results for major language pairs.

The authors address several limitations of the existing RNN architectures that "require a lengthy walk-through, word by word, of the entire input sentence and limit parallelization capability". At the same time, any proposed architecture needs "a wide context window to fully encode the input sentence content". As such, the Transformer network architecture is designed "solely on attention mechanisms, dispensing with recurrence and convolutions entirely". The authors achieved state-of-the-art performance on English to German and French translations, and significantly cut down training costs, compared to those of the best models from the literature.

Approach
---
(NMT encoder-decoder)
The Transformer model utilizes the competitive Encoder-Decoder structure as described by the authors below. In layman terms, the encoder network abstracts the input sentence into a concise neural embeddings that fed into the decoder network to produce translation output. Usually, the encoder is a bi-directional RNN to produce the hidden states forward and backward. The decoder output a translated sentence word-by-word that maximize the softmax probabilities.

> The encoder maps an input sequence of symbol representations (x1; :::; xn) to a sequence of continuous representations z = (z1; :::; zn). Given z, the decoder then generates an output sequence (y1; :::; ym) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

In previous literature, the attention mechanism is a must-have addition to gain sufficient performance on NMT encoder-decoder system. It makes use of the association between each particular input word with the next output word, just like how a real human translator has to look back and forth between source sentence and what outputs have been translated. The diagram describe the following process:
* The input sentence is converted to word embeddings, based on a look-up vocabulary matrix
* From the input-word embeddings, a bi-directional RNN is used to encode the contextual meaning of the input sentence, in a sequential, backward and forward fashion. That way, the hidden states
of the encoder network account for the left and right context of the input sentence
	* Imagine the sentence "I arrived at the bank after crossing the river". To encode the word "bank" properly, we need to read the sentence backward. However, a sequential RNN could only determine that “bank” is likely to refer to the bank of a river after reading each word between “bank” and “river” step by step. This is one of the caveats of RNN architectures and tend to create failures and speed deficiency for long sequences
* Then, we compute the hidden states of the **decoder** network based on 3 inputs **at every time step**
	* The previous hidden state $s_{i-1}$
	* The previous **output** word $EMB_{y_{i-1}}$. If it is the start of the translated sentence, the word token will be <start>
	* The context vector $c_i$ at that time step. This is the "attention" mechanism constructed by $c_i = \sum \alpha_{ij} h_{j}$, which is the weighted contribution of each input word $j$ with regards to the previous hidden state that encode what already been translated
* After the current hidden state $s_i$ is computed by $s_i = f(s_{i-1}, EMB_{y_{i-1}}, c_i)$, non-linear function $f$ can be RELU/tanh, it is fed in as input to the next time step
* Also at time step $i$, we can predict the translated word by feeding the $f(s_{i-1}, EMB_{y_{i-1}}, c_i)$ to a softmax distribution
* The cost for each output word is the negative log of the probability given to the correct word translation: $-log t_i [y_i]$

![](attention.png)

Algorithm
---
The traditional "attention" mechanism, as discussed above, computes the association between an output word with each input word. In contrast, the self-attention mechanism as proposed in this paper computes the association between **any input word and any other input word**. This method "refines the representation of each input word by enriching it with context words that help to disambiguate it" according to Koehn (2017). Each self-attention unit for a sequence of vectors $h_j$ is packed into a matrix H as:

SA = softmax$(\frac{H H^T}{\sqrt{|h|}})H$

The $H H^T$ is sort of like the covariance-variance matrix computation. "The resulting vector of normalized association values is used to weigh the context words". Self-attention layers are more optimal than traditional recurrent and convolutional layers in a few aspects:
* Total computational complexity per layer
* Amount of computation that can be parallelized
* Ability of encode the long-range dependencies in the network. This is especially for translating long sequences where in the traditional recurrent model, the forward and backward signals have to traverse the network. Self-attention units shorten the paths "between any combination of positions in the input and output sequences"
* Self-attention units can be analyzed to decode how "individual attention heads learn to perform different tasks" and yield more interpretable models

![](transformer.png)

Evaluation & Future Work
---
The Tranformer network is evaluated on the English-German translation task. It is trained on 4.5 million sentence pairs from WMT 2014 with a 37,000 vocabulary size (vector length of each word embedding). The final BLEU score is 28.4, more than 2.0 higher than the best previously reported models. Training took 3.5 days on eight P100 GPUs. Clearly, the network "surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models". The model also yields a state-of-the-art BLEU score of 41.0 on the WMT 2014 English-to-French translation task.

For experiments on English constituency parsing, "a 4-layer Transformer model of depth 1024 is trained on the Wall Street Journal (WSJ) portion of the Penn Treebank, about 40K training sentences". The evaluations suggest that "despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models". 

The evaluations demonstrate the effectiveness of Transformer network with self-attention units. The network reduces the training cost and complexity and is designed to scale and transfer to other NLP tasks. The authors release the codebase called [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) in order to accelerate research and extend Transformer application to image, audio, and video.

Reflection
---
This paper introduces the Transformer model and the Tensor2Tensor repository that have been my core research at Boeing. Given the success of this model on general-domain translation tasks, I have been running experiments to evaluate their effectiveness on specific-domain NLP tasks and occasionally finetune the network on aviation-domain corpus. The model is relatively easy to work with and this paper aids my mathematical understanding and provides guidance on how to customize my network to output precise predictions.

Although revolutionary, the Transformer model as proposed in this paper still has a long journey for the reseach community to finetune and customize training for other NLP tasks. Whether it can remain the state-of-the-art network for multiple NLP task is still an ongoing debate. Recently, Google published [BERT](https://github.com/google-research/bert) which is a spin-off from Transformer and claimed that the ImageNet moment has finally arrived for NLP. 

References
---
- [Neural Machine Translation by Philipp Koehn](https://arxiv.org/abs/1709.07809)
- [Google AI blog on Transformer network](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)