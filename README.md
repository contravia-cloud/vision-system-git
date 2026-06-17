# vision-system-git
  - isaac-sim sample code and document
  - ZED camera test 


## isaac-sim docker 실행
```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    --user $(id -u):$(id -g) \
    -e "PRIVACY_CONSENT=Y" \
    -v $HOME/.Xauthority:/isaac-sim/.Xauthority \
    -e DISPLAY \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -u 1234:1234 \
    nvcr.io/nvidia/isaac-sim:5.1.0
```

## isaacsim-webrtc-streaming-client 실행
```bash
./isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage --no-sandbox
```

## 환경 구성 확인
python3.10 가상환경 구성  
cd lecture-coordinate-validation/  
python3.10 validate_stage4.py  
확인  
cd lecture-yolo-segmentation/  
. run_training.sh  

## 예제실행
cd lecture-standalone  
. ~/isaac-sim/python.sh lecture_01_basic_environment.py



## 용어 정리
  - ONNX(Open Neural Network Exchange), 온닉스
    PyTorch, TensorFlow 등 다양한 머신러닝/딥러닝 프레임워크 간에 모델을 자유롭게 교환하고 사용할 수 있도록 만들어진 개방형 표준 파일 포맷
  - TensorRT
    NVIDIA가 만든 딥러닝 모델 추론(Inference) 가속화 라이브러리입니다. 쉽게 말해, 이미 학습이 완료된 인공지능 모델을 NVIDIA GPU에서 가장 빠르고 효율적으로 실행할 수 있도록 '재구성'해주는 최적화 엔진
    보통은 'PyTorch 모델 \(\rightarrow \) ONNX 변환 \(\rightarrow \) TensorRT 최적화'의 과정을 거칩


