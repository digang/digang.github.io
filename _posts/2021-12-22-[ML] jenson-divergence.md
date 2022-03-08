---
layout: post
title : "[ML 머신러닝] Jenson-Shannon-divergence"
categories: [ML, 머신러닝, Jenson-Shannon-divergence]
tags: [entropy, ml, 머신러닝, 다이벌전스,JSD] # TAG 는 소문자로 작성할 것
use_math : true
---

# **Jensen-Shannon Divergence를 왜 사용하는가**

오늘은 약간의 **KL-divergence**의 복습과 '**Jensen-Shannon Divergence**'에 대해서 알아보도록 하겠다.

---

## **KL-divergence?**

아마 누군가는 이렇게 생각할 수도 있다. '이미 KL-divergence'가 존재하는데도 불구하고 왜 또다른 개념이 나왔을까? 그 이유와 과정에 대해 먼저 짚고 넘어가자.

지난 시간에 포스팅했지만, KL-divergence는 두 확률분포의 유사도를 나타내는 값이라 할 수 있다. 즉 **P가 실제 target값 Q가 예측값으로 실제와 예측값의 차이가 얼마나 괴리감이 있는가?**에 대한 척도를 나타내는 값이였다. 하지만 많은 사람들이 착각하는것이, 'KL-divergence'는 두 확률분포의 거리를 의미한다라고 생각하는데, **전혀 아니다**. 

- 일단 KL-divergence 는 symmetric 하지 않다. 즉 대칭성을 띄지 않는다.
- 예를들어 $D_{KL} (P \parallel Q) \ne D_{KL} (Q\parallel P)$이다.
- 거리의 값을 의미한다면 두개의 값은 같아야한다.

---

## **Jenson-Shannon-Divergence**

이러한 부분을 보완하기위해 <span class="custom_underline">**거리개념 Distance Metric**</span>으로 쓸 수 있는 방법에 대해 나온것이 <span class="custom_underline">**Jenson-Shannon-Divergence**</span>이다.

식을 먼저 살펴보면 다음과 같다.

$$
JSD(P,Q) = \frac{1}{2} D(P\parallel M) + \frac{1}{2} D(Q\parallel M)
$\$

$$
where M = \frac{1}{2} (P+Q)
$\$

- 식을 살펴보면, M은 P와 Q의 중간값을 의미한다.
- 각각 M과 KL-divergence를 함으로써 값이 Symmetry해짐을 알 수 있다.

**따라서 다음이 성립한다.**

$$
JSD(P,Q) = JSD(Q,P)
$\$

<span class="custom_underline_green">**이를 통하여 우리는 두 확률분포 사이의 거리(distance)를 JSD를 통해 척도로 사용이 가능해짐을 알 수 있다.**</span>

---
## **Reference**

[https://aigong.tistory.com/66](https://aigong.tistory.com/66) - 아이공의 AI 공부 도전기