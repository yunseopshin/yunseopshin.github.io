---
layout: single
title: 'Laplace Approximation for neural networks'
categories: deep_learning
tag: [딥러닝, 불확실성, python]
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---


### 목차

- Introduction
- Laplace Approximation
- Approximation of Hessian matrix
- Summary

원래 방법론은 VI(BBP), MC-DropOut, SGLD, pSGLD 이 정도만 다뤄보려 했지만 다른 리서치 논문이나 리뷰 논문들을 읽어봤을 때 이 외에도 Laplace Approximation을 통해 사후분포를 추정하려는 시도도 있는 것 같아 이를 짧게 다뤄보려 한다.

## 1. Introduction

데이터 $\mathcal{D}$가 주어졌을 때 모델 parameter $\theta$ 의 사후분포를 구하는 과정을 생각해보자. 이전에는 VI나 MCMC를 베이스로 해 사후분포를 근사했다면, 이번에는 조금 다른 관점에서 문제를 바라본다. 사후분포를 평균을 MAP로 가지고, 분산은 사후분포의 observed fisher information으로 가지는 정규분포로 근사시킨다. 이는 베이즈이론에서 중심극한정리와 비슷한 역할을 하는 Bernstein–von Mises theorem에 의해 정당화 될 수 있다.

## 2. Laplace Approximation

$\hat{\theta} = \text{argmax}_{\theta} p(\theta \mid \mathcal{D})$ 라 하자. $\hat{\theta}$은 보통의 neural network에서 SGD를 사용해 찾을 수 있다. 이 상황에서 $\log p(\theta \mid \mathcal{D})$의 $\theta = \hat{\theta}$에서의 이차 테일러 전개를 생각해보자.

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) + (\theta - \hat{\theta})^{t} \frac{\partial}{\partial \theta} \log p(\hat{\theta} \mid \mathcal{D}) + \frac{1}{2} (\theta - \hat{\theta})^{t} \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} \mid \mathcal{D})(\theta - \hat{\theta})
$$

여기서 $\hat{\theta}$가 MAP이므로 $\frac{\partial}{\partial \theta} \log p(\hat{\theta} \mid \mathcal{D}) =0$ 이 되므로 ,

$$
\begin{aligned}
\log p(\theta \mid \mathcal{D}) &\approx \log p(\hat{\theta} \mid \mathcal{D}) + \frac{1}{2} (\theta - \hat{\theta})^{t} \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} \mid \mathcal{D})(\theta - \hat{\theta}) \\
&= p(\hat{\theta} \mid \mathcal{D}) - \frac{1}{2} (\theta - \hat{\theta})^{t} \left[- \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} \mid \mathcal{D}) \right] (\theta - \hat{\theta})
\end{aligned}
$$

이 된다. 여기서 양변에 밑이 $e$인 지수를 취하면 $p(\theta \mid \mathcal{D})$는 평균이 $\hat{\theta}$이고 분산이 $\bar{H} = \left[- \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} \mid \mathcal{D}) \right] ^{-1}$ 인 정규분포로 근사됨을 알 수 있다.


## 3. Approximation of Hessian matrix

$\hat{\theta}$ 은 neural network에서 SGD를 통해 학습 가능하지만 이 경우 $\bar{H}$ 이걸 구하는게 문제가 된다. $\bar{H}$ 를 전개해서 써보면 다음과 같다.

$$
\begin{aligned}
    \bar{H} = - \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} \mid \mathcal{D}) = - \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} ) - \sum_{i=1}^{N} \frac{\partial^{2}}{\partial \theta^{2}} \log p(y_{i} \mid x_{i}, \hat{\theta})
\end{aligned}
$$

여기서 문제가 생기는데 일단 $\theta$의 차원이 $d$라 할때 이것을 저장하는데만 $O(d^{2})$ 의 메모리가 필요하며, 계산시간은 $O(d^{3})$에 비례하는데 이때 딥러닝에 적용하면 $d$가 매우 켜져 계산상의 이슈가 발생한다. 이를 극복하기 위해 2번의 근사과정이 진행된다.

첫째는 Fisher information 의 성질

$$
\mathbb{E} \left[- \frac{\partial^{2}}{\partial \theta^{2}} \log p(y \mid x, \hat{\theta})\right] = \mathbb{E} \left[ \left( \frac{\partial}{\partial \theta} \log p(y \mid x, \hat{\theta})  \right) \left( \frac{\partial}{\partial \theta} \log p(y \mid x, \hat{\theta})  \right)^{t} \right]
$$

를 사용해 근사를 진행한다.

그 뒤에는 행렬 전체의 성분을 사용하지 않고 대각성분만 사용해 근사를 진행한다. 그 결과는 다음과 같다.

$$
\bar{H} \approx - \frac{\partial^{2}}{\partial \theta^{2}} \log p(\hat{\theta} ) + \sum_{i=1}^{N} \text{Diag} \left( \frac{\partial}{\partial \theta} \log p(y_{i} \mid x_{i}, \hat{\theta}) \right)^{2}
$$

이것을 사용해 역행렬을 구할 때 계산시간은 $O(d)$에 비례해 훨씬 효율적이라고 한다.

추론과정은 이전 방법론들과 마찬가지로 $\theta \sim N(\hat{\theta}, \bar{H})$ 로 추출하여 Monte Calro를 사용해 추정한다.

## 4. Summary

Laplace Approximation 자체는 베이즈 통계에서 1991년에 나온 방법론이다. 오늘은 그 방법을 어떻게하면 parameter의 개수가 많은 neural network에 적용시킬 수 있을지에 대한 아이디어에 대해 알아 봤다. (Ritter, 2018)에는 이보다 더 자세히 Kronecker factorized, curvature, regularization 등에 관한 내용도 나와있는데 당장에는 메인 아이디어만 알고 가면 된다고 생각했기에 이것까지는 깊게 다루진 않았다. 궁금한 독자들은 해당 논문을 찾아보면 좋을 것 같다.

내 생각에 이 모델이 다른 모델에 비해 가지는 장점은 일단 MAP 학습하는 과정이 기존의 SGD를 사용하면 된다는 것에서 다른 모델들과 차별화 되는 장점을 가지는 듯 싶다. 또한 SGLD와 BBP와 달리 이 모델은 학습과정는 표본추출 과정을 필요로 하지 않기에 학습시간에서 장점이 있을걸로 생각된다.

## 5. Reference

    1. Ritter, Hippolyt, Aleksandar Botev, and David Barber. "A scalable laplace approximation for neural networks." 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings. Vol. 6. International Conference on Representation Learning, 2018.
    2. Kim, Minyoung, and Timothy Hospedales. "BayesDLL: Bayesian Deep Learning Library." arXiv preprint arXiv:2309.12928 (2023).
