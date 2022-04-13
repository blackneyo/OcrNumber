# OcrNumber
숫자손글씨minist데이터셋



minist dataset 사용

classes = 10개씩

뱃치 사이즈 = 128 (가중치 128)

에포치 = 10회 (반복학습 10회)

![tk_1](https://user-images.githubusercontent.com/87853267/163077323-78fdac74-9118-446a-870e-532b54ecccb1.png)
![tk_2](https://user-images.githubusercontent.com/87853267/163077329-1316300c-91d1-4a5c-8b9c-b35b7fcf4ba8.png)


인식률이 100퍼센트가 넘어가는 걸 보면

모델의 파라미터 값 수정이 좀 필요할 듯


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
