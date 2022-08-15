---
layout : post
categories: [Python, 파이썬, module, collections]
title : "[Python] - 코테 쓸만한 함수, 모듈 정리"
tags: [백준, 알고리즘, 파이썬, python] # TAG 는 소문자로 작성할 것
use_math : true
---

# **목록**

1. Counter
2. combinations
3. permutations
4. upper
5. isalpha

---

## **1. Counter**

> Counter 는 Collections 모듈에 들어있는 클래스로써 다음과 같이 선언할 수 있다.

```python
from collections import Counter
```

이때, Counter(대문자)임에 유의하도록 하자.

다음은 **Counter** 의 함수들에 대해 알아보도록 하자.

### **elements()**

```python
import collections

ex_counter = collections.Counter("I want success")
print(list(ex_counter.elements()))
print(sorted(ex_counter.elements()))    # 정렬

>>> ['I', ' ', ' ', 'w', 'a', 'n', 't', 's', 's', 's', 'u', 'c', 'c', 'e']
>>> [' ', ' ', 'I', 'a', 'c', 'c', 'e', 'n', 's', 's', 's', 't', 'u', 'w']
```

입력된 값의 요소를 풀어서 반환한다. 반환되는 요소는 무작위로 반환된다! 또한, 요소의 총 개수가 1보다 작을 시 반환하지 않는다.

`elements()`는 단순히 요소를 풀어서 출력해주며 대소문자를 구분한다!

`sorted()`로 오름차순 정렬을 해주었다.

### **most_common(n)**

```python
import collections

ex_counter = collections.Counter(['kim', 'kim', 'park', 'choi', 'kim', 'kim', 'kim', 'choi', 'park', 'choi'])
print(ex_counter.most_common())
print(ex_counter.most_common(2))
print(ex_counter.most_common(1))

>>> [('kim', 5), ('choi', 3), ('park', 2)]
>>> [('kim', 5), ('choi', 3)]
>>> [('kim', 5)]
```

`most_common(n)`함수는 입력된 값의 요소들 중 빈도수(최빈값)을 n개 반환한다. 최빈값을 반환하므로 빈도수가 높은 순으로 상위 n개를 반환하며, 결과값은 Tuple자료형이다.

`most_common()` 빈 값이면 요소 전체를 반환한다.
`most_common(2)` 최빈값 상위 2개를 반환한다.
`most_common(n)` 최빈값 상위 1개를 반환한다.

### **substract()**

```python
import collections

ex_counter1 = collections.Counter('I love you')
ex_counter2 = collections.Counter('I love my family')
# 2번 카운터 - 1번 카운터
ex_counter2.subtract(ex_counter1)
print(ex_counter2)

>>> Counter({'m': 2, ' ': 1, 'l': 1, 'y': 1, 'f': 1, 'a': 1, 'i': 1, 'I': 0, 'v': 0, 'e': 0, 'o': -1, 'u': -1})
```

`substact()` 는 말 그대로 요소를 빼준다. 만약, 요소가 없는 경우인데 `subtract()`를 진행했다면 음수의 값이 반환된다.

이와 더불어, **Counter** 클래스는 <span class="custom_underline">**산술, 집합연산이 가능하다.**</span> 이것이 정말 강력한 기능이다.

---

## **2. combinations**

추가 예정.