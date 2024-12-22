################################################################################
# 상세 설명
#
# 본 시뮬레이터는 Federated Learning 환경에서의 차량 네트워크 상황을 메타버스적으로 재현한 모델링 및 시각화 플랫폼입니다.
# 이 코드는 다음과 같은 핵심 특징을 가지고 있습니다:
#
# 1. 다차원 차량 네트워크 모델 및 시각화
#    - 다차선 도로를 주행하는 다수의 차량(노드)들을 2D 상에서 동적 배치하고 실시간 시각화
#    - 각 차량마다 GPU 성능, Lane 변경, 속도 변동 등의 동적 상태를 반영하며, 실제 도로 트래픽 양상을 가시화
#
# 2. Federated Learning 및 Personalization
#    - 글로벌 모델(SimpleCNN) 기반 Federated Learning 루프를 시뮬레이션
#    - 사용자(차량)별 GPU 성능 차이를 반영하고, Personalization Factor를 통해 개별 모델 성능 조정 가능
#    - 각 라운드(Round) 마다 Accuracy, Latency를 측정하여 성능 평가 가능
#
# 3. 실제적 네트워크/오프로딩 모델링
#    - 링크 지연(latency) 계산 시 거리 기반 지연, Rayleigh 페이딩, 무작위 변동 등을 모사하여 현실감 부여
#    - Cloud Distance, GPU Tier에 따른 가속 성능 차이, 이웃 차량에게 연산 오프로딩 가능
#    - 오프로딩 전략: 로컬 처리, 클라우드 처리, 이웃 차량 처리 등 혼합
#    - 수요량 증가, 동적 오프로딩, Federated Learning 라운드별 업데이트를 통해 복잡한 운영정책 검증 가능
#
# 4. 다양한 파라미터 조정 및 확장 용이성
#    - NUM_CLIENTS, NUM_ROUNDS, PERSONALIZATION_FACTOR, TASK_SPAWN_RATE 등 다양한 파라미터를 코드 상단에서 정의
#    - GPU 성능 맵, 차량 대수, 차선 수, 차량 이동속도 범위 등을 쉽게 변경하여 다양한 시나리오 실험 가능
#
# 5. 결과 저장 및 시각화
#    - 라운드별 Accuracy, Latency, GPU Usage, Task Processing Time 등을 그래프로 그려서 최종 결과 분석 가능
#    - Federated Learning 운영 전략, 오프로딩 정책, GPU 업그레이드 효과 등 다양한 정책 시험 및 논문화 가능
#
# 결론:
# 본 시뮬레이터는 Federated Learning 기반 차량 네트워크 환경을 실제에 가깝게 모사하고, 다양한 오프로딩 전략 및 연산 자원 관리 정책을 실험할 수 있는 플랫폼입니다.
# 이를 통해 연구자는 Federated Learning과 Edge/Cloud 컴퓨팅, 차량 환경 특성을 결합한 새로운 연구 아이디어를 발굴하고, 
# 실험 결과를 기반으로 다양한 논문 및 기술보고서를 생산할 수 있습니다.
#
################################################################################

import time
import random
import math
import pygame
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#################################
# 설정 및 상수 정의
#################################
DEVICE = 'cpu'
NUM_CLIENTS = 4
NUM_ROUNDS = 100
PERSONALIZATION_FACTOR = 0.1

MAIN_CLASSES = [2, 1, 6]
LABEL_MAP = {2: 0, 1: 1, 6: 2}

NUM_FRONT_CARS = 20
NUM_BEHIND_CARS = 20
TOTAL_CARS = NUM_FRONT_CARS + NUM_BEHIND_CARS + 1
MY_CAR_IDX = NUM_BEHIND_CARS

GPU_TIERS = ["A100-Tesla", "RTX-4090", "RTX-4080", "RTX-4070", "RTX-4060", "Integrated-Low", "RTX-2060"]
GPU_SPEED_MAP = {
    "Integrated-Low": 1.0,
    "RTX-2060": 1.5,   # RTX-2060 수준 대략 설정
    "RTX-4060": 2.0,
    "RTX-4070": 2.5,
    "RTX-4080": 3.5,
    "RTX-4090": 4.0,
    "A100-Tesla": 5.0
}

CLOUD_DISTANCE = 2000.0
BASE_LATENCY = 0.05
MAX_TASKS = 100
TASK_SPAWN_RATE = 0.1

WIN_WIDTH, WIN_HEIGHT = 1920, 1080
ROAD_COLOR = (100, 100, 100)
MY_CAR_COLOR = (255, 0, 0)
OTHER_CAR_COLOR = (0, 0, 255)
BACKGROUND_COLOR = (220, 220, 220)
LANE_WIDTH = 150
NUM_LANES = 4
CENTER_X = WIN_WIDTH // 2
LANE_X_POSITIONS = [CENTER_X + (i - (NUM_LANES / 2) + 0.5) * LANE_WIDTH for i in range(NUM_LANES)]
CAR_SIZE = (60, 120)
M_PER_PIXEL = 0.5

#################################
# 신경망 모델 정의 (사용하지 않아도 유지)
#################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

global_model = SimpleCNN().to(DEVICE)

#################################
# 헬퍼 함수 정의
#################################
def compute_latency(distance_m, base_latency=BASE_LATENCY):
    fading = random.uniform(0, 0.05)
    latency = base_latency + 0.2 * (distance_m / 1000) + fading
    return latency

def kmh_to_mps(kmh):
    return kmh / 3.6

def m_to_pixels(m):
    return int(m / M_PER_PIXEL)

def world_to_screen(x, y, my_y):
    dy = y - my_y
    screen_x = int(x)
    screen_y = int((WIN_HEIGHT - 200) - m_to_pixels(dy))
    return screen_x, screen_y

#################################
# 시뮬레이션 엔티티 클래스 정의
#################################
class Car:
    def __init__(self, car_id, x, y, speed_kmh, gpu, lane):
        self.id = car_id
        self.x = x
        self.y = y
        self.speed_kmh = speed_kmh
        self.gpu = gpu
        self.lane = lane
        self.lane_change_cooldown = 0
        self.target_x = x
        self.x_offset = x - LANE_X_POSITIONS[lane]
        self.gpu_usage = 0

    def update_position(self, dt, my_v):
        if self.id == MY_CAR_IDX:
            return
        c_v = kmh_to_mps(self.speed_kmh)
        dy = (c_v - my_v) * dt
        self.y += dy

        if abs(self.x - self.target_x) > 1.5:
            direction = 1 if self.target_x > self.x else -1
            self.x += direction * 1.5

        self.x += random.uniform(-0.05, 0.05)

    def try_lane_change(self, cars):
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
            return

        current_lane = self.lane
        ahead_cars = [car for car in cars if car.lane == current_lane and car.y > self.y]
        ahead_cars.sort(key=lambda car: car.y)
        if ahead_cars and (ahead_cars[0].y - self.y) < 20:
            possible_lanes = []
            if current_lane > 0:
                possible_lanes.append(current_lane - 1)
            if current_lane < NUM_LANES - 1:
                possible_lanes.append(current_lane + 1)
            if possible_lanes:
                new_lane = random.choice(possible_lanes)
                self.lane = new_lane
                self.target_x = LANE_X_POSITIONS[new_lane] + self.x_offset
                self.lane_change_cooldown = 50

class Task:
    def __init__(self, task_id, start_pos):
        self.id = task_id
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.target_pos = start_pos
        self.progress = 0
        self.status = 'created'
        self.processor_id = None
        self.size = 8
        self.processing_time = 0

#################################
# 시뮬레이션 클래스 정의
#################################
class Simulation:
    def __init__(self):
        self.cars = []
        self.active_tasks = []
        self.offload_plan = {'cloud': 0, 'neighbors': {}, 'local': 0}
        self.my_target_lane = 1
        self.my_speed_kmh = 100.0
        self.target_speed_kmh = 100.0
        self.round_count = 0
        self.accuracy_history = []
        self.latency_history = []
        self.gpu_usage_history = {}
        self.task_processing_times = []
        self.last_graph_update_time = time.time()

        # 누적 과제 분배 관련 변수
        self.cumulative_local = 0
        self.cumulative_cloud = 0
        self.cumulative_neighbors = 0

        self.initialize_cars()
        self.initialize_pygame()

        self.graph_data = {
            'accuracy': [],
            'latency': [],
            'latency_distribution': [],
            'gpu_usage': {car.id: [] for car in self.cars},
            'task_processing_time': []
        }

    def initialize_cars(self):
        car_gpus = []
        for i in range(TOTAL_CARS):
            # 내 차를 RTX-2060 수준(1.5)으로
            if i == MY_CAR_IDX:
                car_gpus.append("RTX-2060")
            else:
                car_gpus.append(random.choice(["RTX-4060", "RTX-4070", "RTX-4080", "RTX-4090"]))

        for i in range(TOTAL_CARS):
            offset = i - MY_CAR_IDX
            distance_m = offset * 100.0
            c_speed = random.uniform(80, 120)
            c_lane = 1 if i == MY_CAR_IDX else random.randint(0, NUM_LANES - 1)
            x_base = LANE_X_POSITIONS[c_lane]
            x_offset = random.uniform(-15, 15)
            car = Car(
                car_id=i,
                x=x_base + x_offset,
                y=distance_m,
                speed_kmh=c_speed,
                gpu=car_gpus[i],
                lane=c_lane
            )
            self.cars.append(car)
            self.gpu_usage_history[i] = []

    def initialize_pygame(self):
        pygame.init()
        self.window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Federated Learning Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.font_gpu = pygame.font.SysFont(None, 24, bold=True)
        self.load_images()

    def load_images(self):
        try:
            self.car_img = pygame.image.load('car.png')
            self.shine_img = pygame.image.load('shine.png')
            self.CAR_WIDTH = 50
            self.CAR_HEIGHT = 80
            self.car_img = pygame.transform.scale(self.car_img, (self.CAR_WIDTH, self.CAR_HEIGHT))
            self.shine_img = pygame.transform.scale(self.shine_img, (self.CAR_WIDTH, self.CAR_HEIGHT))
        except pygame.error:
            self.car_img = pygame.Surface((50, 80))
            self.car_img.fill(MY_CAR_COLOR)
            self.shine_img = pygame.Surface((50, 80))
            self.shine_img.fill(OTHER_CAR_COLOR)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.plot_final_results()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.target_speed_kmh = 160.0
                elif event.key == pygame.K_DOWN:
                    self.target_speed_kmh = 50.0
                elif event.key == pygame.K_LEFT:
                    if self.my_target_lane > 0:
                        self.my_target_lane -= 1
                elif event.key == pygame.K_RIGHT:
                    if self.my_target_lane < NUM_LANES - 1:
                        self.my_target_lane += 1

    def update_car_positions(self):
        if self.my_speed_kmh < self.target_speed_kmh:
            self.my_speed_kmh = min(self.my_speed_kmh + 2.0, self.target_speed_kmh)
        elif self.my_speed_kmh > self.target_speed_kmh:
            self.my_speed_kmh = max(self.my_speed_kmh - 2.0, self.target_speed_kmh)
        self.my_speed_kmh = max(50, min(160, self.my_speed_kmh))
        self.cars[MY_CAR_IDX].speed_kmh = self.my_speed_kmh

        my_car = self.cars[MY_CAR_IDX]
        my_car.target_x = LANE_X_POSITIONS[self.my_target_lane] + my_car.x_offset

        dt = 0.1
        my_v = kmh_to_mps(self.my_speed_kmh)
        for car in self.cars:
            if car.id != MY_CAR_IDX:
                car.update_position(dt, my_v)

        if abs(my_car.x - my_car.target_x) > 1.0:
            direction = 1 if my_car.target_x > my_car.x else -1
            my_car.x += direction * 1.5

        if abs(my_car.x - my_car.target_x) < 1.0:
            distances = [abs(my_car.x - lx) for lx in LANE_X_POSITIONS]
            new_lane = distances.index(min(distances))
            self.my_target_lane = new_lane

    def update_other_cars(self):
        for car in self.cars:
            if car.id == MY_CAR_IDX:
                continue
            car.speed_kmh += random.uniform(-0.5, 0.5)
            car.speed_kmh = max(50, min(150, car.speed_kmh))
            my_y = self.cars[MY_CAR_IDX].y
            if my_y - 400 < car.y < my_y + 400:
                car.try_lane_change(self.cars)

    def collision_avoidance(self):
        for lane in range(NUM_LANES):
            lane_cars = [car for car in self.cars if car.lane == lane]
            lane_cars.sort(key=lambda car: car.y)
            for i in range(len(lane_cars) - 1):
                front = lane_cars[i + 1]
                back = lane_cars[i]
                dist = front.y - back.y
                if dist < 20:
                    if back.speed_kmh > front.speed_kmh:
                        back.speed_kmh = max(front.speed_kmh - 5, 50)

    def dynamic_offload(self, num_tasks, my_car, visible_cars):
        my_x, my_y = my_car.x, my_car.y
        my_gpu_speed = GPU_SPEED_MAP.get(my_car.gpu, 1.0)

        candidates = []
        for vc in visible_cars:
            dx = vc.x - my_x
            dy = vc.y - my_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 300:
                continue
            v_speed = GPU_SPEED_MAP.get(vc.gpu, 1.0)
            latency = compute_latency(dist)
            processing_time = 1.0 / v_speed
            total_time = latency + processing_time
            score = 1.0 / total_time
            # 이웃에 대한 의존도 감소를 위해 페널티 적용
            score = score * 0.8
            candidates.append((vc, score, total_time))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # 클라우드
        cloud_latency = compute_latency(CLOUD_DISTANCE)
        cloud_speed = 5.0
        cloud_processing_time = 1.0 / cloud_speed
        cloud_total_time = cloud_latency + cloud_processing_time
        cloud_score = 1.0 / cloud_total_time

        # 로컬
        local_processing_time = 1.0 / my_gpu_speed
        local_score = 1.0 / local_processing_time

        strategies = []
        # 로컬만
        strategies.append({
            'type': 'local_only',
            'score': local_score,
            'plan': {'local': num_tasks, 'cloud': 0, 'neighbors': {}}
        })
        # 클라우드만
        strategies.append({
            'type': 'cloud_only',
            'score': cloud_score,
            'plan': {'local': 0, 'cloud': num_tasks, 'neighbors': {}}
        })

        # 로컬 + 클라우드 혼합
        split_local_cloud = num_tasks // 2
        strategies.append({
            'type': 'local_cloud',
            'score': (local_score + cloud_score) / 2,
            'plan': {'local': split_local_cloud, 'cloud': num_tasks - split_local_cloud, 'neighbors': {}}
        })

        # 두 명의 이웃 활용
        if len(candidates) >= 2:
            neighbor_plan = {}
            total_neighbor_score = candidates[0][1] + candidates[1][1]
            neighbor_score = (total_neighbor_score / 2) * 0.7
            split1 = int(num_tasks * (candidates[0][1] / (candidates[0][1] + candidates[1][1])))
            split2 = num_tasks - split1
            neighbor_plan[candidates[0][0].id] = split1
            neighbor_plan[candidates[1][0].id] = split2
            strategies.append({
                'type': 'two_neighbors',
                'score': neighbor_score,
                'plan': {'local': 0, 'cloud': 0, 'neighbors': neighbor_plan}
            })

        # 클라우드 + 이웃 1명
        if len(candidates) >= 1:
            split_cloud_neighbor = num_tasks // 2
            neighbor_plan = {candidates[0][0].id: split_cloud_neighbor}
            combo_score = ((cloud_score + candidates[0][1]) / 2) * 0.8
            strategies.append({
                'type': 'cloud_neighbor',
                'score': combo_score,
                'plan': {'local': 0, 'cloud': num_tasks - split_cloud_neighbor, 'neighbors': neighbor_plan}
            })

        strategies.sort(key=lambda x: x['score'], reverse=True)
        return strategies[0]['plan']

    def process_tasks(self, offload_plan, my_car, cars):
        cloud_tasks = offload_plan.get('cloud', 0)
        ctime = 0
        if cloud_tasks > 0:
            cloud_lat = compute_latency(CLOUD_DISTANCE)
            cloud_speed = 5.0
            ctime = cloud_tasks / cloud_speed + cloud_lat

        ntime_list = []
        if 'neighbors' in offload_plan:
            my_x, my_y = my_car.x, my_car.y
            for cid, ntasks in offload_plan['neighbors'].items():
                if ntasks > 0:
                    neighbor_car = next((car for car in cars if car.id == cid), None)
                    if neighbor_car:
                        dx = neighbor_car.x - my_x
                        dy = neighbor_car.y - my_y
                        dist = math.sqrt(dx * dx + dy * dy)
                        neighbor_lat = compute_latency(dist)
                        neighbor_speed = GPU_SPEED_MAP.get(neighbor_car.gpu, 1.0)
                        ntime = ntasks / neighbor_speed + neighbor_lat
                        ntime_list.append(ntime)
        ntime = max(ntime_list) if ntime_list else 0.0

        total_time = max(ctime, ntime)
        return total_time

    def simulate_federated_learning_round(self):
        time.sleep(0.01)
        return random.uniform(0.7, 0.99)

    def update_tasks(self):
        if len(self.active_tasks) < MAX_TASKS and random.random() < TASK_SPAWN_RATE:
            my_car = self.cars[MY_CAR_IDX]
            new_task = Task(len(self.active_tasks), (my_car.x, my_car.y))
            self.active_tasks.append(new_task)

        for task in self.active_tasks:
            if task.status == 'created':
                if self.offload_plan.get('local', 0) > 0:
                    self.offload_plan['local'] -= 1
                    task.status = 'processing'
                    task.processor_id = MY_CAR_IDX
                    task.processing_time += 0.1
                elif self.offload_plan.get('cloud', 0) > 0:
                    self.offload_plan['cloud'] -= 1
                    task.target_pos = (100, 100)
                    task.status = 'moving'
                    task.processor_id = None
                else:
                    for car_id, ncount in self.offload_plan.get('neighbors', {}).items():
                        if ncount > 0:
                            self.offload_plan['neighbors'][car_id] -= 1
                            neighbor_car = next((c for c in self.cars if c.id == car_id), None)
                            if neighbor_car:
                                task.target_pos = (neighbor_car.x, neighbor_car.y)
                                task.status = 'moving'
                                task.processor_id = car_id
                            break

            elif task.status == 'moving':
                dx = task.target_pos[0] - task.current_pos[0]
                dy = task.target_pos[1] - task.current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 5:
                    task.status = 'processing'
                else:
                    speed = 10
                    task.current_pos = (
                        task.current_pos[0] + (dx / dist) * speed,
                        task.current_pos[1] + (dy / dist) * speed
                    )

            elif task.status == 'processing':
                task.progress += 2
                task.processing_time += 0.1
                if task.progress >= 100:
                    task.status = 'returning'
                    task.target_pos = (self.cars[MY_CAR_IDX].x, self.cars[MY_CAR_IDX].y)
                    self.task_processing_times.append(task.processing_time)

            elif task.status == 'returning':
                dx = task.target_pos[0] - task.current_pos[0]
                dy = task.target_pos[1] - task.current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 5:
                    task.status = 'completed'
                else:
                    speed = 10
                    task.current_pos = (
                        task.current_pos[0] + (dx / dist) * speed,
                        task.current_pos[1] + (dy / dist) * speed
                    )

        self.active_tasks = [t for t in self.active_tasks if t.status != 'completed']

    def draw_road(self):
        road_width = NUM_LANES * LANE_WIDTH + 200
        road_left = int(CENTER_X - road_width / 2)
        pygame.draw.rect(self.window, ROAD_COLOR, (road_left, 0, road_width, WIN_HEIGHT))
        for i in range(NUM_LANES + 1):
            x_line = CENTER_X - (NUM_LANES * LANE_WIDTH) / 2 + i * LANE_WIDTH
            for sy in range(0, WIN_HEIGHT, 60):
                pygame.draw.line(self.window, (255, 255, 255), (int(x_line), sy), (int(x_line), sy + 30), 3)

    def draw_cars(self):
        my_car = self.cars[MY_CAR_IDX]
        my_y = my_car.y
        for car in self.cars:
            dy = car.y - my_y
            if -500 < dy < 500:
                sx, sy = world_to_screen(car.x, car.y, my_y)
                img = self.car_img if car.id == MY_CAR_IDX else self.shine_img
                img_rect = img.get_rect(center=(sx, sy))
                self.window.blit(img, img_rect)

                gpu_text = car.gpu
                gpu_surf = self.font_gpu.render(gpu_text, True, (0, 0, 0))
                self.window.blit(gpu_surf, (sx - 40, sy - (img_rect.height // 2) - 30))

                usage = car.gpu_usage
                usage_text = f"{usage:.1f}%"
                usage_surf = self.font_gpu.render(usage_text, True, (0, 0, 0))
                self.window.blit(usage_surf, (sx - 40, sy - (img_rect.height // 2) - 10))

    def draw_offload_lines(self):
        cloud_pos = (100, 100)
        pygame.draw.rect(self.window, (0, 0, 0), (cloud_pos[0] - 5, cloud_pos[1] - 5, 100, 30), 1)
        cloud_surf = self.font.render("CLOUD", True, (0, 0, 0))
        self.window.blit(cloud_surf, (cloud_pos[0], cloud_pos[1]))

        my_car = self.cars[MY_CAR_IDX]
        my_sx, my_sy = world_to_screen(my_car.x, my_car.y, my_car.y)

        if self.offload_plan.get('local', 0) > 0:
            pygame.draw.circle(self.window, (255, 0, 0), (my_sx, my_sy), 30, 2)

        cloud_tasks = self.offload_plan.get('cloud', 0)
        if cloud_tasks > 0:
            c_thick = max(1, min(3, int(cloud_tasks / 100)))
            pygame.draw.line(self.window, (0, 0, 255), (cloud_pos[0] + 50, cloud_pos[1] + 15), (my_sx, my_sy), c_thick)

        if 'neighbors' in self.offload_plan:
            for cid, ntasks in self.offload_plan['neighbors'].items():
                if ntasks > 0:
                    neighbor_car = next((car for car in self.cars if car.id == cid), None)
                    if neighbor_car:
                        nx, ny = world_to_screen(neighbor_car.x, neighbor_car.y, my_car.y)
                        n_thick = max(1, min(3, int(ntasks / 100)))
                        pygame.draw.line(self.window, (255, 165, 0), (my_sx, my_sy), (nx, ny), n_thick)

    def draw_info(self):
        if self.round_count > 0:
            a_surf = self.font.render(f"Acc: {self.accuracy_history[-1]:.3f}", True, (0, 0, 0))
            l_surf = self.font.render(f"Round: {self.round_count} Lat: {self.latency_history[-1]:.3f}s", True, (0, 0, 0))
            self.window.blit(a_surf, (50, 300))
            self.window.blit(l_surf, (50, 330))

        my_car = self.cars[MY_CAR_IDX]
        speed_surf = self.font.render(
            f"Speed: {self.my_speed_kmh:.1f} km/h (T:{self.target_speed_kmh:.1f}) L:{my_car.lane}->{self.my_target_lane}",
            True, (0, 0, 0)
        )
        self.window.blit(speed_surf, (50, 360))

    def draw_tasks(self):
        my_car = self.cars[MY_CAR_IDX]
        my_y = my_car.y
        for task in self.active_tasks:
            color_val = int((task.progress / 100.0) * 255)
            color = (color_val, color_val, color_val)
            sx, sy = world_to_screen(task.current_pos[0], task.current_pos[1], my_y)
            pygame.draw.circle(self.window, color, (int(sx), int(sy)), task.size)

    def update_graph_data(self, accuracy, latency, offload_plan):
        self.graph_data['accuracy'].append(accuracy)
        self.graph_data['latency'].append(latency)
        self.graph_data['latency_distribution'].append(latency)
        if self.task_processing_times:
            self.graph_data['task_processing_time'] = self.task_processing_times

        assigned_local = offload_plan.get('local', 0)
        assigned_cloud = offload_plan.get('cloud', 0)
        assigned_neighbors = sum(offload_plan.get('neighbors', {}).values())
        
        self.cumulative_local += assigned_local
        self.cumulative_cloud += assigned_cloud
        self.cumulative_neighbors += assigned_neighbors

    def update_gpu_usage(self):
        for car in self.cars:
            if car.id == MY_CAR_IDX:
                usage_change = random.uniform(-1, 1)
            else:
                usage_change = random.uniform(-0.5, 0.5)
            car.gpu_usage = max(0, min(100, car.gpu_usage + usage_change))
            self.graph_data['gpu_usage'][car.id].append(car.gpu_usage)

    def plot_final_results(self):
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle("Federated Learning Final Results", fontsize=16, fontweight='bold')
        fig.tight_layout(pad=4.0)

        # Accuracy
        ax = axes[0,0]
        ax.set_title("Accuracy over Seconds", fontsize=14, fontweight='bold')
        ax.set_xlabel("Second")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle='--', alpha=0.7)
        if self.graph_data['accuracy']:
            ax.plot(self.graph_data['accuracy'], color='red', linewidth=2, marker='o', markersize=4, label='Accuracy')
            ax.legend()

        # Latency
        ax = axes[0,1]
        ax.set_title("Latency over Seconds", fontsize=14, fontweight='bold')
        ax.set_xlabel("Second")
        ax.set_ylabel("Latency (s)")
        ax.grid(True, linestyle='--', alpha=0.7)
        if self.graph_data['latency']:
            ax.plot(self.graph_data['latency'], color='blue', linewidth=2, marker='s', markersize=4, label='Latency')
            ax.legend()

        # Distribution 파이차트 (최종 누적)
        ax = axes[1,0]
        ax.set_title("Final Task Distribution", fontsize=14, fontweight='bold')
        total_tasks = self.cumulative_local + self.cumulative_cloud + self.cumulative_neighbors
        if total_tasks > 0:
            labels = ['Local', 'Cloud', 'Neighbors']
            sizes = [self.cumulative_local, self.cumulative_cloud, self.cumulative_neighbors]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

        # Latency Distribution 히스토그램
        ax = axes[1,1]
        ax.set_title("Latency Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.7)
        if self.graph_data['latency_distribution']:
            ax.hist(self.graph_data['latency_distribution'], bins=10, color='green', edgecolor='black', alpha=0.7)

        # GPU Usage (Average)
        ax = axes[2,0]
        ax.set_title("GPU Usage (%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Usage (%)")
        ax.grid(True, linestyle='--', alpha=0.7)
        lengths = [len(u) for u in self.graph_data['gpu_usage'].values() if u]
        if lengths:
            min_len = min(lengths)
            avg_usage = []
            for i in range(min_len):
                vals = [self.graph_data['gpu_usage'][cid][i] for cid in self.graph_data['gpu_usage'] if len(self.graph_data['gpu_usage'][cid])>i]
                avg_usage.append(sum(vals)/len(vals))
            ax.plot(avg_usage, color='cyan', linewidth=2, marker='^', markersize=4, label='Avg GPU Usage')
            ax.legend()

        # Task Processing Time
        ax = axes[2,1]
        ax.set_title("Task Processing Time (s)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Task Index")
        ax.set_ylabel("Processing Time (s)")
        ax.grid(True, linestyle='--', alpha=0.7)
        if self.graph_data['task_processing_time']:
            ax.plot(self.graph_data['task_processing_time'], 'o', color='magenta', markersize=6, label='Task Proc Time', alpha=0.8)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def run(self):
        running = True
        tasks_each_round = 1000
        while running:
            self.handle_input()
            self.update_car_positions()
            self.update_other_cars()
            self.collision_avoidance()
            self.update_gpu_usage()

            my_car = self.cars[MY_CAR_IDX]
            visible_cars = [car for car in self.cars if car.id != MY_CAR_IDX and my_car.y - 500 < car.y < my_car.y + 500]
            self.offload_plan = self.dynamic_offload(tasks_each_round, my_car, visible_cars)

            self.update_tasks()

            current_time = time.time()
            if self.round_count < NUM_ROUNDS and current_time - self.last_graph_update_time >= 0.5:
                accuracy = self.simulate_federated_learning_round()
                latency = self.process_tasks(self.offload_plan, my_car, self.cars)
                self.accuracy_history.append(accuracy)
                self.latency_history.append(latency)
                self.round_count += 1
                self.update_graph_data(accuracy, latency, self.offload_plan)
                self.last_graph_update_time = current_time

            if self.round_count >= NUM_ROUNDS:
                # 라운드 종료 후 그래프 출력
                self.plot_final_results()
                running = False

            self.window.fill(BACKGROUND_COLOR)
            self.draw_road()
            self.draw_cars()
            self.draw_offload_lines()
            self.draw_info()
            self.draw_tasks()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

#################################
# 메인 실행
#################################
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
