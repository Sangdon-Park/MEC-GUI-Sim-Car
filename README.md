# Federated Learning 기반 차량 네트워크 시뮬레이터

본 레포지토리는 **Federated Learning 환경에서의 차량 네트워크**를 메타버스 관점에서 모사하고,  
**다차선 도로에서 주행 중인 차량들과 FL 라운드, 오프로딩(Offloading) 전략, GPU 자원 사용** 등을  
실시간으로 시뮬레이션하며 시각화하는 통합 플랫폼입니다.

---

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능](#주요-기능)  
3. [시스템 요구사항](#시스템-요구사항)  
4. [설치 및 실행 방법](#설치-및-실행-방법)  
5. [코드 구조](#코드-구조)  
    - [1) 전역 변수 및 상수](#1-전역-변수-및-상수)  
    - [2) 핵심 클래스](#2-핵심-클래스)  
    - [3) 주요 함수 및 로직](#3-주요-함수-및-로직)  
6. [사용 방법 안내 (단축키)](#사용-방법-안내-단축키)  
7. [시뮬레이션 데이터 수집 및 결과 확인](#시뮬레이션-데이터-수집-및-결과-확인)  
8. [확장 포인트 및 응용](#확장-포인트-및-응용)  
9. [라이선스](#라이선스)  

---

## 프로젝트 개요
본 시뮬레이터는 **차량 네트워크**를 다차선 도로 위에 재현하고, 각 차량(노드)이 **Federated Learning**(FL) 모델에 참여하면서  
**클라우드, 이웃 차량, 로컬** 중 어느 곳에 연산을 오프로딩(offload)할지 동적 결정하는 과정을 시각화합니다.

- 차량은 다차선 환경에서 속도를 변경하거나 차선을 바꾸는 행위를 수행  
- 각 차량은 GPU 성능이 다르며, 이웃 차량에게 오프로딩할 때 거리 기반 지연(latency)과 Rayleigh 페이딩 등을 고려  
- **Federated Learning** 루프가 라운드 단위로 진행되며, 라운드마다 차량의 로컬 모델 훈련 상태(accuracy)를 업데이트  
- 오프로딩에 따른 **태스크 처리 시간(task processing time)** 및 **네트워크 지연(latency)**를 추적  
- 게임엔진(`pygame`)을 사용하여 **실시간**으로 차선, 차량, 오프로딩 링크 등을 시각화  

---

## 주요 기능
1. **다차선 도로와 차량 움직임 시각화**  
   - 최대 4차선 도로에서 다수 차량을 2D 형태로 배치, 실제 주행처럼 속도/차선 변경  
   - 차선 유지 혹은 추월(차선변경) 등 동적인 차량 거동 모델링  

2. **Federated Learning 시뮬레이션**  
   - 라운드 수(`NUM_ROUNDS`)만큼 연속 진행되며, 각 라운드마다 무작위 accuracy 측정(예시)  
   - SimpleCNN 모델 클래스를 예시로 포함(실제 학습 없이 확장 가능)  
   - FL 라운드 진행 시 accuracy, latency가 리스트로 기록되어 결과 시각화에 활용  

3. **Dynamic Offloading**  
   - 로컬 처리, 클라우드 처리, 이웃 차량 처리 등 여러 오프로딩 전략을 동적으로 탐색  
   - 거리 기반 레이턴시(`compute_latency()`)와 GPU 성능 맵(`GPU_SPEED_MAP`)을 활용해 오프로딩 최적화  
   - 각 시뮬레이션 스텝마다 최적 전략을 선정 후 offload_plan에 반영  

4. **실시간 태스크 관리**  
   - 태스크(`Task`)가 생성되면, 오프로딩 계획에 따라 로컬/클라우드/이웃차량으로 전송  
   - 프로세싱이 끝나면(진행도 100%), 원점으로 결과를 돌려보내 완료 상태가 됨  
   - 태스크별 처리 시간, 거리에 따른 이동/전송 시뮬레이션 포함  

5. **데이터 로깅 및 최종 결과 플롯**  
   - FL 라운드 종료(`NUM_ROUNDS` 도달) 시, matplotlib로 결과 그래프(Accuracy, Latency 등) 자동 생성  
   - 최종 태스크 분포(로컬/클라우드/이웃) 파이차트, Latency 히스토그램, GPU 사용량 그래프 등 다양한 통계 지표 시각화  

---

## 시스템 요구사항
- **Python 3.7 이상**  
- `pygame`  
- `torch` (PyTorch)  
- `matplotlib`  

설치 예시:
```bash
pip install pygame torch matplotlib
```
(추가적으로 CUDA 환경이 있으면 GPU 학습 테스트 가능하지만, 기본적으로 CPU 환경으로도 동작)

---

## 설치 및 실행 방법
1. 본 저장소를 클론하거나 ZIP 파일로 다운로드  
2. `simulation.py` (또는 본 `.py` 파일)와 동일한 디렉토리에서 다음 명령 실행:
    ```bash
    python simulation.py
    ```
3. 시뮬레이션 창이 열리며, **차선, 차량, 오프로딩 링크** 등이 실시간으로 표현됩니다.  

---

## 코드 구조

### 1) 전역 변수 및 상수
- **NUM_CLIENTS**, **NUM_ROUNDS**: Federated Learning에 참여하는 (가정상) 클라이언트 수, FL 라운드 수  
- **GPU_SPEED_MAP**: GPU 모델별 처리 속도를 가중치 형태로 매핑 (예: A100-Tesla = 5.0)  
- **LANE_WIDTH, NUM_LANES**: 도로 차선 폭, 차선 수  
- **MAIN_CLASSES, LABEL_MAP**: 예시로 사용되는 클래스 라벨 정보  
- **TASK_SPAWN_RATE, MAX_TASKS**: 태스크가 생성될 확률, 최대 태스크 수  

### 2) 핵심 클래스
1. **`SimpleCNN`**  
   - PyTorch 기반의 예시 신경망 구조 (Federated Learning 실험용)  
   - 실제 학습 로직은 (본 예제에서는) 무작위 Accuracy로 대체  

2. **`Car`**  
   - 각 차량의 **위치(x, y)**, **속도(speed_kmh)**, **GPU 모델**, **차선** 등을 보관  
   - **lane_change_cooldown**(차선 변경 후 일정 시간 유지) 로직, **update_position()** 메서드로 위치 갱신  
   - 가까운 차량과의 간격이 좁으면 차선 변경을 시도하는 등 간단한 행동 규칙이 정의  

3. **`Task`**  
   - 태스크 생성 시 시작 위치를 가지고, 오프로딩 대상(로컬/클라우드/이웃차량)에 따라 이동->처리->결과 반환 과정 모델링  
   - **progress**, **status**(created, moving, processing, returning, completed) 등을 통해 태스크 생애주기 관리  

4. **`Simulation`**  
   - **pygame** 기반 메인 루프(while running)에서 이벤트 처리, 차량 위치 업데이트, 태스크 업데이트, FL 라운드 관리 등을 총괄  
   - `dynamic_offload()`를 통해 오프로딩 전략 결정 -> `process_tasks()`에서 실제 처리 시간 추정  
   - `simulate_federated_learning_round()`에서 임의 Accuracy 값을 생성(현실적 확장은 사용자가 커스텀 가능)  
   - 모든 데이터(`accuracy_history`, `latency_history`, `task_processing_times`, `gpu_usage` 등)를 모아 최종 그래프 생성  

### 3) 주요 함수 및 로직
- **`compute_latency(distance_m, base_latency=BASE_LATENCY)`**  
  - 거리(m)에 비례한 추가 지연과 랜덤 페이딩을 합산해 레이턴시 계산  
- **`dynamic_offload(num_tasks, my_car, visible_cars)`**  
  - 로컬 / 클라우드 / 이웃 차량 후보 중에서 가장 좋은 조합을 찾아서 task 분배 계획(딕셔너리) 생성  
- **`update_tasks()`**  
  - 태스크 상태(machine -> moving -> processing -> returning -> completed) 진행도 업데이트  
  - 오프로딩 계획(offload_plan)에 따라 태스크를 적절히 배정  
- **`simulate_federated_learning_round()`**  
  - 실제론 PyTorch 모델 업데이트가 이뤄지겠지만, 여기서는 임의 Accuracy(0.7~0.99 범위) 반환  
- **`plot_final_results()`**  
  - 라운드 완료 시(`round_count >= NUM_ROUNDS`) Accuracy, Latency, 오프로딩 파이차트, Latency 히스토그램, GPU Usage, Task Processing Time 등 6개 서브플롯을 생성하여 시각화  

---

## 사용 방법 안내 (단축키)
- **상/하 키(↑/↓)**: 현재 차량의 목표 속도를 크게 높임(160km/h) / 낮춤(50km/h)  
- **좌/우 키(←/→)**: 차량 차선 변경 (왼쪽 또는 오른쪽 차선으로 이동 시도)  
- **종료**: 시뮬레이션 창에서 닫기 버튼(ESC는 별도 정의 없음)  

실행 후, 2D 화면에서 **내 차량**(빨간색)과 **이웃 차량**(파란색)을 확인할 수 있으며,  
차선표시 / 오프로딩 라인 / 태스크(원형) / GPU 사용량(차량 위 텍스트) 등이 시각화됩니다.

---

## 시뮬레이션 데이터 수집 및 결과 확인
- 시뮬레이션이 진행되며, 특정 시간(기본 0.5초 간격)마다 FL 라운드가 진행되어 **accuracy**와 **latency**가 기록됩니다.  
- **NUM_ROUNDS**에 도달하면 시뮬레이션이 자동으로 정지,  
  `plot_final_results()`가 호출되어 **matplotlib 그래프(6개 서브플롯)** 창이 팝업으로 표시됩니다.  
- 각 그래프는:
  - Accuracy, Latency(시계열)
  - 로컬/클라우드/이웃 차량 오프로딩 분포(파이차트)
  - Latency 히스토그램
  - GPU Usage 추이
  - Task Processing Time  
  등으로 구성되어, Federated Learning + 오프로딩의 성능 변화를 한눈에 파악할 수 있습니다.

---

## 확장 포인트 및 응용
1. **실제 데이터 학습**  
   - `simulate_federated_learning_round()` 부문의 임의 accuracy 대신, PyTorch 모델을 실제로 학습시키고 통계를 반영하도록 확장 가능  
2. **오프로딩 정책 고도화**  
   - 현재는 단순한 거리-기반 지연 + GPU 성능만 고려하지만,  
     동적 트래픽, 대역폭 제한, 지연 보증(QoS) 등을 추가하여 더욱 정교한 오프로딩 로직을 설계 가능  
3. **개인화(Personalization) 적용**  
   - 차량마다 모델이 조금씩 다른 Personalization Factor를 적용해 FL 라운드별로 성능 편차 관찰  
4. **도로/차량 모델링 확장**  
   - 고속도로/도심지 시뮬레이션, 교차로, 신호등 등의 복잡한 교통 시나리오도 추가 가능  
5. **에너지 소비 모델**  
   - GPU 연산 시 전력 소비, 배터리 잔량, 각 차량의 에너지 효율 등을 고려한 종합 모델로 확장 가능  

---

## 라이선스
- 본 프로젝트는 자유로운 **연구/교육 목적** 사용을 권장합니다.  
- 외부 라이브러리(pygame, torch, matplotlib)는 각각의 라이선스를 따릅니다.  
- 프로젝트 개선이나 버그 리포트, Pull Request 등은 언제든지 환영합니다.
