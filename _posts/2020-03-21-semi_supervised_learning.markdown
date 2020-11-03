---
layout: draft
title: "Semi-supervised learning"
permalink: /semi_supervised_learning
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

## Basic solution

#### Normalization flow-based

Train:

$$
L(\theta) = \sum_{(x_{i},y_{i}) \in labelled} log p_{\theta} (x_{i}, y_{i}) + \sum_{x_{j} \in unlabelled} log p_{\theta}(x_{j})
$$

$$p_{\theta} (x_{i}, y_{i})$$ can be solved by(1). _Hybrid models_[^hybridmodel] or (2). further decomposed as$$
p_{\theta} (x_{i}, y_{i}) = p_{\theta}(y_{i}) * p_{\theta}(x_{i}|y_{i})
$$

(1). Hybrid model decompose
$$
p_{\theta} (x, y)
$$  to  $$
p_{\theta}(x)*p_{\theta}(y|x)
$$

 This paper  use Generalized Linear Models(GLM) to model 
 $$
 p(y_{i}|x_{i})
 $$, P(x) is modeled by Normalization Flow.

(2). $$
p(x)
$$ is modeled by normalization flow, $$
p(y|x)
$$ is modeled by conditional normalization flow(novelty lies in).


Testing:
$$p(y|x) = p(x,y)/p(x)$$

## Semi-supervised image classification

Summmary from FixMatch:

![](/imgs/fixmatch-compare.png)



**[Label Propagation for Deep Semi-supervised Learning,CVPR19](https://arxiv.org/abs/1904.04717)**

![](/imgs/lp.png)


**[Realistic Evaluation of Deep Semi-Supervised Learning Algorithms,NeuIPS18](https://arxiv.org/pdf/1804.09170.pdf)**


**[Building One-Shot Semi-supervised (BOSS) Learning up to Fully Supervised Performance,Arxiv2006](https://arxiv.org/pdf/2006.09363.pdf)**

motivated by barely-supervised learniing in FixMatch.

A combination of fixmatch, self-training, etc.




**[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence,Arxiv2001](https://arxiv.org/pdf/2001.07685.pdf)**

![](/imgs/fixmatch.png)

- weak-augmentation-> only flip-and-shift data augmentation.
- stron-augmentation-> Inspired
by UDA [45] and ReMixMatch [2], we leverage CutOut
[13], CTAugment [2], and RandAugment [10] for strong
augmentation, which all produce heavily distorted versions
of a given image.
- Why thresholding?Inspired
by UDA [45] and ReMixMatch [2], we leverage CutOut
[13], CTAugment [2], and RandAugment [10] for strong
augmentation, which all produce heavily distorted versions
of a given image.
- basic experimental choices that are
often ignored or not reported when new SSL methods are
proposed **(such as the optimizer or learning rate schedule)**
because we found that they can have an outsized impact on
performance.
- weak,weak doesn't work; weak,strong works.
- The loss is composed of supervised loss and unsupervised loss. the weight of unsupervised loss should increase gradually by annealing.
- Detail matters, weight decay regularization is particularly important.
- SGD is better than Adam. Table 5.
- use a cosine learning rate decay,Table 6.
- In summary, we observe that swapping pseudo-labeling for
sharpening and thresholding would introduce a new hyperparameter while achieving no better performance.
- Of the aforementioned work, FixMatch bears the closest
resemblance to two recent algorithms: Unsupervised Data
Augmentation (UDA) [45] and ReMixMatch [2].
- .............  We find that to obtain strong results, especially in
the limited-label regime, certain design choices are often
underemphasized – most importantly, weight decay and the
choice of optimizer. The importance of these factors means
that even when controlling for model architecture as is recommended in [31], the same technique can not always be
directly compared across different implementations.



**[Rethinking Semi–Supervised Learning in VAEs,Arxiv2006](https://arxiv.org/pdf/2006.10102.pdf)**

**[Semisupervised knowledge transfer for deep learning from private training data.ICLR17 best paper]()**

relate SSL and differential privacy


**[MixMatch: A Holistic Approach to Semi-Supervised Learning,NeuIPS19](https://arxiv.org/pdf/1905.02249.pdf)**

![](/imgs/mixmatch.png)
![](/imgs/mixmatch0.png)

- Table 4 is full of information.
- The relation of privacy-preserving is also interesting.
- finicky point:mixup has max function to gurantee the mixed result biases towards labelled image x.
- finicky point： X use shannon-entropy, U use MSE.


**[Temporal Ensembling for Semi-Supervised Learning,ICLR17](https://arxiv.org/pdf/1610.02242.pdf)**

![](/imgs/temporal-ensembling.png)


$$z_{i}$$ is $$N \times C$$, will moving averaged to $$\tilde{z_{i}}$$, check alg 1 in the paper.



**[Mean Teacher,Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,NIPS17](https://arxiv.org/abs/1703.01780)**

Motivated by Temporal Emsembling. Temporal Emsembling is moving averaged on output, mean teacher is moving averaged on network parameters.

Teacher model is the moving average of student model, do not reverse.

![](/imgs/mean-teacher.png)

mean squared error (MSE) as our consistency cost function, MSE is better than KL-divergence experimentally.

Three different noises $$\upeta$$ are considered: The model architecture is a 13-layer convolutional neural network (ConvNet) with three types of noise: random translations and horizontal flips of the input images, Gaussian noise on the input layer, and dropout applied within the network.


**CutOut**

**CutMix**

## Semi-supervised semantic segmentation

**[Virtual Adversarial Training:
A Regularization Method for Supervised and
Semi-Supervised Learning,PAMI17](https://arxiv.org/pdf/1704.03976.pdf)**

- interesting and simple idea, high citation.
  
![](/imgs/vat.png)

sceenshot from [S4L: Self-Supervised Semi-Supervised Learning,ICCV19](https://arxiv.org/pdf/1905.03670.pdf)


**Consistency regularization**

Consistency regularization (Sajjadi et al., 2016b; Laine & Aila, 2017; Miyato et al., 2017; Oliver
et al., 2018) describes a class of semi-supervised learning algorithms that have yielded state-ofthe-art results in semi-supervised classification, while being conceptually simple and often easy to
implement. The key idea is to encourage the network to give consistent predictions for unlabeled
inputs that are perturbed in various ways.

**[Semi-Supervised Semantic Segmentation with Cross-Consistency Training,CVPR20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ouali_Semi-Supervised_Semantic_Segmentation_With_Cross-Consistency_Training_CVPR_2020_paper.pdf)**

- pertubation function is important
- Semi-Supervised Domain Adaptation, and combintation with image-level labels is a bonus
  

**[Adversarial Learning for Semi-Supervised Semantic Segmentation,BMVC18](https://arxiv.org/pdf/1802.07934.pdf)**

![](/imgs/adv-semi-seg.png)

Pay more attention to $$L_{semi}$$ loss, by thresholding the output of discriminator network to construct a psydo label.


**[CowMix, Semi-supervised semantic segmentation needs strong, high-dimensional perturbations ](https://openreview.net/forum?id=B1eBoJStwr)**

> The use of a rectangular mask restricts the dimensionality of the perturbations that CutOut and
CutMix can produce. Intuitively, a more complex mask that has more degrees of freedom should
provide better exploration of the plausible input space. We propose combining the semantic CutOut
and CutMix regularizers introduced above with a novel mask generation method, giving rise to two
regularization methods that we dub CowOut and CowMix due to the Friesian cow -like texture of
the masks.

## Semi-supervised detection

**[Consistency-based Semi-supervised Learning for Object Detection,NeurIPS19](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection)**


**[A Simple Semi-Supervised Learning Framework for Object Detection,Arxiv2005](https://arxiv.org/pdf/2005.04757.pdf)**


#### Footnotes
* footnotes will be placed here. This line is necessary
{:footnotes}

[^hybridmodel]: Hybrid Models with Deep and Invertible Features, ICML19.




