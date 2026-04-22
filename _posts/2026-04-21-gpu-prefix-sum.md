---
layout: post
title: "GPU prefix sum"
date: 2026-04-21
categories: Algorithm
tags: ["prefix sum, Koggle-stone, Blelloch"]
--- 

GPU에서의 prefix sum (누적합)을 계산하는 방법을 알아보자.

prefix sum, cummulative sum, inclusive scan 혹은 간단히 scan은 **배열의 첫 번째 원소부터 특정 위치(인덱스)까지의 원소들을 차례대로 더한 값을 미리 계산하여 저장해두는 알고리즘 기법**으로, 

두 수열 {${x_1, x_2, … x_n}$} 과 {${y_1, y_2, … y_n}$} 에 대해 $y_i = \sum\limits_{k=1}^{i}x_k$를 만족한다.

Prefix sum은 간단히 $y_i = y_{i-1} + x_i$과 같은 점화식을 통해 구할 수 있으며, Prefix sum 배열을 만드는데는 $O(n)$의 시간복잡도가 소요되나, 1 ~ i 번째 원소들의 합은 $O(1)$라는 시간복잡도로 구할 수 있다.

이러한 Prefix sum은 CPU에서는 간단히 구현 할 수 있지만, 병렬처리를 하는 GPU에서는 이야기가 달라진다.

이 글에서는 GPU에서 병렬적으로 prefix sum을 구하는 2가지 방법을 소개한다.

들어가기 전, 2가지 용어를 소개한다.

### Terminology

- Inclusive scan : Prefix sum 배열의 원소 $y_i$가 $y_i = \sum\limits_{k=1}^i x_k$를 만족한다.
- Exclusive scan : Prefix sum 배열의 원소 $y_i$가 $y_i = \sum\limits_{k=1}^{i-1} x_k$를 만족한다. 즉, i번째 인덱스의 원소는 제외한다.

## 1. Koggle-Stone scan (Naive approach)

아래는 Koggle-Stone scan을 도식화 한 것이다.

![koggle_stone.png]({{ site.baseurl }}/assets/images/blelloch/kogglestone.png)

(출처 : https://github.com/shineyruan/CUDA-Stream-Compaction)

부분 누적합들을 서로 더해가며 전체 누적합을 구하는 방식으로,

$k ≥ 2^{d-1}$ 인 $x_k$에 대해 $x[k] = x[k - 2^{d-1}] + x[k]$를 총 $log_2n$번 수행한다.

총 연산량은 $O(nlogn)$이고, 시간 복잡도(병렬)은 $O(logn)$이다.

최종 결과에서 기존 배열의 원소를 빼면 Exclusive scan 결과를 얻을 수 있다.

- 장점 :
    1. **최저 지연 시간 (Minimum Latency):** 연산 깊이가 ⌈log2n⌉로, 병렬 Prefix sum 알고리즘 중 가장 빠르다.
    2. **낮은 분기 분기 (Low Branch Divergence):** 스레드들이 수행하는 작업이 정형화되어 있어 `if-else`에 의한 성능 저하가 적다.
    3. **높은 SIMD 효율성:** 각 단계마다 대부분의 스레드가 활성화되어 연산을 수행하므로, GPU의 Warp 내 연산 유닛(ALU) 활용도가 극대화된다.
- 단점 :
    1. **높은 작업 복잡도 (Work Inefficient):** 전체 연산 횟수가 $O(n log n)$ 이다. 이후 설명할 Blelloch scan ($O(n)$) 에 비해 절대적인 연산량 자체가 많아 전력 소모가 크다.

Kogge-Stone scan은 많은 연산량 때문에 꼭 필요한 상황이 아니라면 잘 사용하지 않는다.

## 2. Blelloch scan (Work-efficient approach)

Blelloch scan은 Up-sweep(Reduce) 단계와 Down-sweep 단계로 이루어져 있다.

다음은 Up-sweep 단계를 도식화한 것이다.

![blelloch_up.png]({{ site.baseurl }}/assets/images/blelloch/blellochup.png)

(출처 : https://github.com/shineyruan/CUDA-Stream-Compaction)

위 사진이 보여주듯이, 이 알고리즘은 이진트리를 생각하며 이해를 하는 것이 쉽다. (실제 구현은 Inplace 연산이다.)

우선 Up-sweep 단계에서는 위 사진처럼 모든 노드의 합을 구한다. Up-sweep은 간단하므로 Down-sweep으로 넘어가자.

다음은 Down-sweep 단계를 도식화 한 것이다. 

![blelloch_down.png]({{ site.baseurl }}/assets/images/blelloch/blellochdown.png)

(출처 : https://github.com/shineyruan/CUDA-Stream-Compaction)

우선 트리의 루트 노드를 0으로 초기화하는 것으로 시작한다. (이 경우 결과는 exclusive scan을 수행하며, 수열의 첫 원소로 초기화한다면 inclusive scan을 수행한다.)

여기서 **‘부모’의 값은 부모가 루트인 트리의 왼쪽에 있는 값들의 합이다**. 무슨 말인지 모르겠으면, 일단 넘어가자.

그 다음, 부모의 값을 왼쪽 자식과 오른쪽 자식에 ‘**복사**’한다. (더하는 것이 아니다). 그리고 오른쪽 자식에 복사하기 전 왼쪽 자식의 값을 ‘**더한다**’. 

이후, 복사와 더하는 과정을 루트에서 리프 노드로 올라가면서 반복하면, exclusive scan의 결과가 구해진다.

다시 돌아와서 “여기서 ‘부모’의 값은 부모가 루트인 트리의 왼쪽에 있는 값들의 합이다” 이 말을 이해해보자.

![blelloch_scan1.jpeg]({{ site.baseurl }}/assets/images/blelloch/blelloch_scan1.jpeg)

위 사진은 복사와 더하는 과정을 1번 진행한 상황이다. 루트의 오른쪽 자식 6은 6이 루트인 트리(파란 삼각형)보다 왼쪽에 있는 숫자들 (0, 1, 2, 3)의 합과 동일하다. 왼쪽 자식의 경우, 0이 루트인 트리보다 왼쪽에 있는 숫자가 존재하지 않기 때문에 0이 된다. 

리프 노드 4의 입장에서 생각하면,  자신 앞의 숫자 0, 1, 2, 3의 합을 가지고 있으면 된다. 리프 노드 5의 입장에서는 0, 1, 2, 3과 함께 자신의 형제인 4까지의 합이 필요하다.

4와 5의 부모인 9는 그대로 0, 1, 2, 3의 합인 6을 부모에게서 물려받고, 4는 부모가 물려받은 6을 물려받고, 5는 부모가 물려받은 6에 자신의 형제인 4를 더하면 된다.

6과 7의 입장에서는 0, 1, 2, 3에 추가적으로 4와 5까지 더한 값이 필요하다. 그렇기 때문에, 6와 7의 부모인 13은 13의 부모인 6을 물려받되, 형제의 기존 값(9)를 더하여 물려받는다. 이를 통해 6과 7의 부모는 0, 1, 2, 3의 합 (6)과 4와 5의 합 (9)인 15로, 이는 6, 7보다 왼쪽에 있는 값들의 합이 된다.

트리의 루트노드를 트리보다 왼쪽에 있는 값들의 합으로 만들기 위해, 위의 규칙과 같이 알고리즘이 작동한다.

Up-sweep 단계에서 $\sum\limits_{k=1}^{log_2n} \frac{n}{2^k} = n(1-\frac{1}{n}) = n-1$ 번의 연산, Down-sweep 단계에서 $\sum\limits_{k=1}^{log_2n} 2^{k-1} = n-1$ 번의 연산을 하므로, 총 연산량은 $O(n)$이다. 

시간 복잡도(병렬)은 Up-sweep 단계에서 $log_2n$ 단계, Down-sweep 단계에서 $log_2n$ 단계가 소요되므로 $O(logn)$이다.

- 장점 :
    
    **Work-efficient** : 전체 연산 횟수가 적어, 전력 소모가 적고 GPU 스케쥴링 측면에서 유리하다.
    
    **Low computational cost** : 연산 유닛을 적게 사용하기 때문에 자원 효율적이다.
    
- 단점 :
    
    **2-step 구조 :** 구조상 Down-Sweep 단계가 추가되어 Kogge-Stone보다 절대적인 실행 시간(Latency) 자체는 약간 더 길 수 있다.
    

## +) Multi-level Blelloch scan

GPU는 보통 방대한 데이터의 연산을 처리하지만, 위의 두 알고리즘은 아쉽게도 방대한 데이터에 대한 prefix sum을 계산하는데에는 부족하다.

그 이유는 GPU의 성능을 최대한 활용하려면 GPU의 Streaming Multiprocessor(SM)에 있는 공유 메모리(shared memory)를 활용해야 하는데, 이 공유 메모리의 크기가 굉장히 제한적이다.

+) 과거에는 32KB ~ 64KB가 일반적였으나, 최신 GPU들은 더 많이 제공하는 편이다.
    여담으로, 현 시점 가장 큰 shared memory를 제공하는 마이크로아키텍쳐는 NVIDIA Hopper로, Hopper를 사용하는 H100은 227KB를 제공한다.(80GB 한대가 4천만원씩이나 한다..)

또한 그래픽스 API를 사용하는 경우, 어떤 API를 사용하냐에 따라서 할당 받을 수 있는 공유 메모리가 상이하다. DX11의 경우는 쓰레드 그룹당 사용할 수 있는 공유 메모리의 최대 크기는 32KB로 고정되어 있다.

![blelloch3.png]({{ site.baseurl }}/assets/images/blelloch/blelloch3.png)

(출처 : https://dlsys.cs.washington.edu/pdf/lecture5.pdf)

이러한 한계 때문에, Blelloch scan을 여러 단계로 구현한다.

![blelloch_multi.jpeg]({{ site.baseurl }}/assets/images/blelloch/blellochmulti.png)

1. 데이터를 공유 메모리에 적재할 수 있는 크기로 나누어 각 블럭을 적재한다.
2. 각 블럭에 대해 scan을 수행하여 prefix sum을 구한다.
3. 각 블럭에 있는 원소들의 총합 (block sum)을 구하여 보조 배열을 만든다.
4. 이 보조 배열에 대해서 다시 scan을 수행하여 prefix sum을 구한다.
    
    (데이터가 충분히 크다면 3-4의 과정을 반복한다.)
    
5. 구한 block sum들을 각 블록의 local scan 결과에 더해준다.

아래는 multi-level blelloch scan을 수행한 예시이다.

![blellochmulti2.jpeg]({{ site.baseurl }}/assets/images/blelloch/blelloch_scan2.jpeg)

Blelloch scan은 주로 GPU에서의 Radix sort나 stream compaction에 쓰인다.