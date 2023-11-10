# XOR 문제를 해결하기 위한 다층 퍼셉트론을 ReLU와 시그모이드 활성화 함수를 사용하여 구현합니다.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import platform
if platform.system() == 'Darwin': #맥
    plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': #윈도우
    plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
    #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
    #!mv malgun.ttf /usr/share/fonts/truetype/
    #import matplotlib.font_manager as fm 
    #fm._rebuild() 
    plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결
#matplotlib 패키지 한글 깨짐 처리 끝


# ReLU 함수와 그 미분
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 시그모이드 함수와 그 미분
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR 문제에 대한 입력과 출력
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 가중치와 편향 초기화
np.random.seed(0)
weights0 = np.random.uniform(-1, 1, (2, 2))  # 은닉층 가중치
weights1 = np.random.uniform(-1, 1, (2, 1))  # 출력층 가중치
bias0 = np.random.uniform(-1, 1, (1, 2))     # 은닉층 편향
bias1 = np.random.uniform(-1, 1, (1, 1))     # 출력층 편향

# 순전파 함수 정의
def forward(x):
    hidden_layer_input = np.dot(x, weights0) + bias0
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights1) + bias1
    output = sigmoid(output_layer_input)
    return output
    
# 3D 그래프 준비    
fig = plt.figure(figsize=(12, 8))

# 학습률 설정
learning_rate = 0.1

# 학습 과정
n_iterations = 30000
for i in range(n_iterations):
    # 순전파
    hidden_layer_input = np.dot(X, weights0) + bias0
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights1) + bias1
    output = sigmoid(output_layer_input)

    # 역전파
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights1.T)
    hidden_delta = hidden_error * relu_derivative(hidden_layer_output)

    # 가중치와 편향 업데이트
    weights1 += learning_rate * hidden_layer_output.T.dot(output_delta)
    weights0 += learning_rate * X.T.dot(hidden_delta)
    bias1 += learning_rate * np.sum(output_delta, axis=0)
    bias0 += learning_rate * np.sum(hidden_delta, axis=0)

    if i % 100 == 0:
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        
        # meshgrid 생성
        x_range = np.linspace(0, 1, 50)
        y_range = np.linspace(0, 1, 50)
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # meshgrid에서의 예측값 계산 및 평면 시각화
        zz = forward(grid).reshape(xx.shape)
        ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.6)
        
        # XOR 연산의 결과를 3D 공간에 표시
        for j in range(4):
            ax.scatter(X[j, 0], X[j, 1], y[j], color='b' if y[j] == 0 else 'r', s=100)
        
        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_zlabel('Output')
        
        #
        ax.set_xlim((-0.5, 1.5))
        ax.set_ylim((-0.5, 1.5))
        ax.set_zlim((-0.5, 1.5))
        
        #
        title0 = f'3D Plot of XOR Problem at Iteration {i+1}'
        title1 = f'은닉층 가중치, 은닉층 편향 = {weights0}, {bias0}'
        title2 = f'출력층 가중치, 출력층 편향 = {weights1}, {bias1}'
        title = '\n'.join([title0, title1, title2])
        
        #
        ax.set_title(title, wrap=False)
        
        #
        plt.tight_layout()
        plt.pause(0.025)
        plt.savefig('figure_%04d.png' %(i // 100))