
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
weights0 = np.random.random((2, 2))  # 은닉층 가중치
weights1 = np.random.random((2, 1))  # 출력층 가중치
bias0 = np.random.random((1, 2))     # 은닉층 편향
bias1 = np.random.random((1, 1))     # 출력층 편향

# 학습 과정과 시각화를 위한 준비
n_iterations = 10000
frames = 2000
iter_per_frame = n_iterations // frames
plotting_indices = np.linspace(0, n_iterations, frames, endpoint=False, dtype=int)

# 3D 그래프 준비
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 학습 과정
for i in range(n_iterations):
    # 순전파
    hidden_layer_input = np.dot(X, weights0) + bias0
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights1) + bias1
    output = sigmoid(output_layer_input)

    # 역전파
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights1.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # 가중치와 편향 업데이트
    weights1 += hidden_layer_output.T.dot(output_delta)
    weights0 += X.T.dot(hidden_delta)
    bias1 += np.sum(output_delta, axis=0)
    bias0 += np.sum(hidden_delta, axis=0)

    # 시각화 프레임
    if i in plotting_indices:
        ax.clear()
        for j in range(4):
            #ax.scatter(X[j, 0], X[j, 1], y[j][0], color='b' if y[j] == 0 else 'r', s=100)
            #ax.text(X[j, 0], X[j, 1], y[j][0], f"({X[j, 0]}, {X[j, 1]}, {y[j][0]})", size=10, zorder=1)
            ax.scatter(X[j, 0], X[j, 1], output[j][0], color='b' if y[j] == 0 else 'r', s=100)
            ax.text(X[j, 0], X[j, 1], output[j][0], f"({X[j, 0]}, {X[j, 1]}, {output[j][0]})", size=10, zorder=1)

        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_zlabel('Output')

        ax.set_xlim((-1, 2))
        ax.set_ylim((-1, 2))
        ax.set_zlim((-1, 2))

        #
        title0 = f'3D Plot of XOR Problem at Iteration {i+1}'
        title1 = f'은닉층 가중치, 은닉층 편향 = {weights0}, {bias0}'
        title2 = f'출력층 가중치, 출력층 편향 = {weights1}, {bias1}'
        title = '\n'.join([title0, title1, title2])

        #
        ax.set_title(title)

        #
        plt.pause(0.025)

plt.show()
