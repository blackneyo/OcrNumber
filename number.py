from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
'''
to split the data of training and testing sets
훈련세트 테스트세트 데이터 분할/ mnist 데이터 load
2022.04.05 이동한 
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()


'''
모델은 이미지 데이터 직접 가져올 수 없음  ∴기본작업 수행 및 데이터 처리
훈련데이터 3차원 (60000*28*28)
CNN 모델은 한 차원이 더 필요 ∴ 행렬 (60000*28*28*1)
2022.04.05 이동한
'''


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
'''
2022.04.05 이동한
-파라미터-
classes = 10개씩
batch_size = 128 (가중치 128개) 
에포치 = 10회(반복학습 10회)
클래스 백터를 이진 클래스 행렬로 변환
because, 딥러닝의 분류 문제로 인해 원-핫 인코딩 방식을 채용 
∵ 0~9까지 정수 값을 갖는 형태가 아닌 0,1로 이루어진 이진수 백터로 수정해야함
이를 위해 utils의 to_categorical()함수 사용
데이터 프레임 타입 float32로 변경
MNIST의 각각 데이터는 0~255값
/255로 0~1사이의 값으로 변환
∵ 데이터의 분산이 클 때 분산의 정도를 줄여주는 것이 모델의 성능을 높임  -> 정규화(Normalization)
'''
num_classes = 10
batch_size = 128
epochs = 10

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

'''
모델 만들기 2022.04.05 이동한
CNN모델-sequential 

레이어1 conv2D (컨볼루션 레이어)
--필터 32, 필터크기(3,3), relu활성화함수--

레이어2 maxpooling2d (풀링 레이어)
--축소시킬 필터의 크기 (2,2)--
드롭아웃 25% (떨궈내기)

레이어3 flatten (flatten 레이어) 입력 이미지 1차원배열로 반환 -> 간단한 전처리 수행

레이어4 Dense 은닉레이어 뉴런 256개로 구성, relu 활성화 함수 사용
드랍아웃 50%

레이어5 Dense 은닉레이어 뉴런 10개로 구성, softmax 활성화 함수 사용

--학습 과정 설계--

loss = 손실함수/ 모델 훈련시 loss 함수를 최소로 만들어주는 가중치들을 찾는것을 목표로 사용
categorical crossentropy 사용 / 분류해야할 클래스가 3개 이상인 경우, 멀티클래스 분류에 사용
optimizer = 최적화 계획 loss function 에 따라서 network 갱신
Adadelta 알고리즘 구현 옵티마이저 사용/default값 (learning_rate=0.001, rho=0.95, epsilon =1e-07, name='adadelta', **kwargs)
metrics = 평가지표 -> 검승셋과 연관, 훈련 과정을 모니터링 하는데 사용/ overfitting 이나 underfitting 되고 있는 지 여부 확인 
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
# model.summary()

'''
모델 훈련 - 2022.04.05 이동한
keras의 model.fit()함수 호출
훈련데이터, 검증데이터, 배치사이즈, 에포치, 배치크기, verbose=1이면 학습되는 상황을 콘솔창으로 확인 가능
매 에포치마다 손실값과 정확도 -> 모델 튜닝하여 성능 높일 때 사용 
'''

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print("모델 훈련 성공")
model.save('mnist.h5')
print("모델을 mnist.h5로 저장")

'''
모델 평가 - 2022.04.05 이동한
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('testloss', score[0])
print('testaccuracy', score[1])