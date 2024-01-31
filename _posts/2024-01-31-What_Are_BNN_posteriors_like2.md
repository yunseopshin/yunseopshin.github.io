---
layout: single
title: '논문리뷰: What Are Bayesian Neural Network Posteriors Really Like - 2?'
categories: deep_learning
tag: [딥러닝, 불확실성, python]
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---


### 목차

- How well is HMC mixing?
- Background
- Related work
- HMC for deep neural network
- Reference


저번 포스팅에 이어서 읽어보도록 하자. 요즘 허리도 안좋고 그 여파로 발 감각도 이상하고 여간 몸상태가 말이 아니다. 운동 다 열심히 하고 좋은 의자 쓰자. 절실히 느끼는 중이다. 또 그와 별개로 최근에 밴처기업과 테스트 프로젝트하는 것이 있는데 그거 때문에 음성 데이터에 적용되는 transformer 중 하나인 wav2vec2.0을 공부하고 있는데 조금 낯선 내용이다 보니까 이해하기가 어렵네. 추후 해당 논문도 정리해서 포스팅해봐야 겠다.

## 5. How well is HMC mixing?

일단 이 mixing이 뭔지에 대한 문제에 봉착했다. mixture model에서의 혼합을 의미하는 mixing인가? 근데 여기서 그 내용은 크게 상관이 있는것 같지도 않고, 앞부분 내용을 대략 읽어보니 HMC chain의 수렴진단을 어떻게 진행하고 이가 기하적으로 시사하는 바가 뭔지에 대해 알아보는 파트 같은데 해당 용어는 아닌것 같고, 그러면 converging 대신에 mixing을 쓴건가? 이런 생각이 들어 선배한테 여쭤보니, 사후분포가 여러 봉우리(multi modal)인 경우에 각각을 골고루 잘 돌아다닐 수 있는지에 관한 내용이라 한다. 즉, HMC chain이 한곳에서만 계속 왔다갔다 거리는게 아닌 사후분포 전체 영역을 잘 돌아다닐수 있는지에 대한 말로 보면 될 것 같다.

저자들은 weight space와 function space에서의 mixing을 고려하는데, HMC의 경우 function space에서 더 잘 mixing된다는 결론을 내었다.

### 5.1. $\hat{R}$ diagnostics

겔만-루빈(Gelman-Rubin, 1992) 검정은 동일한 사후분포에서 여러개의 체인을 추출했다고 가정하고 진행한다. 전통적인 ANOVA 검정을 이용하여 체인 내의 분산과 체인 간의 분산을 비교해서 체인 간의 분산이 작다고 판단되면 체인이 수렴했다고 판단한다. 이 때 그 판단 기준으로 검정통계량 $\hat{R}$를 사용하는데 이 값이 1에 가까우면 체인이 잘 mixing되고 수렴했다고 판단한다.

저자들은 이것을 표본추출된 weight의 체인 그 자체와, 그것과 dataset을 사용해 softmax를 먹인 output체인에 대해서 $\hat{R}$값을 구해 히스토그램을 그려봤는데 이는 다음과 같다.

![png](/images/What_bnn_files/bnn3.png)

이를 보면 softmax의 결과값으로 나오는 function space에서는 대부분의 chain들에 대해서 $\hat{R}$값이 1.1보다 작음을 확인 가능했다. 이를 통해서 단일 체인이 대부분의 test point의 예측값에 대해서는 다중 체인만큼이나 잘 mixing된다는 걸 확인 가능했다. 이때 weight space에서는 엄청 안좋다라고까진 못해도 몇몇 값들에 대해서는 $\hat{R}$값이 매우 크게 나타나 체인이 mixing에 실패하는 방향이 있음을 알 수 있었다.

### 5.2. Posterior density visualizations

Weight의 사후분포로부터 추출한 체인중 몇 개의 vector만 골라 이중에서도 2개의 성분에 대해서 시각화한 결과이다.

![png](/images/What_bnn_files/bnn4.png)


이때 (a)는 한 체인에 대한 시각화이고 세 시점($t=1,51,101$)을 표시하였고 시각화이고 (b)는 세 체인을 통합한 분포에 대한 시각화이고 각 체인에 대해 동일 시점($t=51$)을 표시하였다. 

(a)에서 보면 각 샘플들은 각각 다른 mode에 위치해 있음을 확인 가능했다. 즉 한 mode에 간 시점이 있더라도, 그 주변에서만 계속 체인이 생기는게 아닌 다른 mode로 이동할 수 있음을 보여준다. 즉 HMC 표본은 parameter space에서 복잡한 형태의 사후분포를 잘 탐험할 수 있음을 확인가능하다. 이는 unimodal 근사법에 비해 확실한 장점이 될 수 있다.

(b)는 (a)에 비해 보다 정규화되있고 대칭적임을 확인가능한데 이는 위에 5.1.에서 언급한 것 처럼 단일 HMC체인이 weight space에서는 완벽하게 mixing되지 않을수 있음을 시사한다.

### 5.3. Convegence of the HMC chains

HMC chain에서 burn-in(번인)의 효과를 살펴본 챕터이다. 초기 표본의 경우 사후분포에서 떨어진 영역을 탐색하고 있어 이 값을 추정에 사용하는 것은 오히려 추정값의 정확성에 문제가 될 수 있다. 이런 경우 초기의 적당한 수의 표본을 버리고 나머지를 가지고 추정하는것을 burn-in(번인)이라고 한다. 이를 CIFAR-10과 IMDB data set에 대해 적용시킨 결과는 다음과 같다.

![png](/images/What_bnn_files/bnn5.png)

이를 통해 두 경우 모두 burn-in의 효과는 어느정도 시점이후에는 미미하고 그렇기 때문에, 저자들은 실험을 진행하는데 있어 50번째 반복을 기준으로 번인을 적용시켰다고 한다.

## 11. Reference

    1. Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. G. (2021, July). What are Bayesian neural network posteriors really like?. In International conference on machine learning (pp. 4629-4640). PMLR.
    2. Aitchison, L. (2020). A statistical theory of cold posteriors in deep neural networks. arXiv preprint arXiv:2008.05912.