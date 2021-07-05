# smilegateAI_1

안녕하세요 🥰

스마일게이트멤버십 AI부문 1기 모집을 지원하는 '우아잘'팀입니다!



사람을 감동시키고 재미있게 만들기 위한 인공지능을 연구해 가면서

요즘 많이 문제로 떠오르고 있는 시험, 면접 부정행위 문제를 해결하여

소중한 열정이 진실 될 수 있도록 하기 위해 '히어캣'서비스를 준비하고 있습니다.



## 🖥 코드 실행 방법

### 1) 모듈 설치

```python
pip install opencv-contrib-python
pip install torch
pip install dlib
pip install face_recognition
pip install easydict
pip install mxnet-cu100
pip install insightface
```



### 2) 실행

실행시키려면,

```bash
# 노트북 웹캠으로 하는 법
python pose_estimation_main.py -

# 파일로 하는 법. 가로세로 종횡비가 4:3 이어야함. 1280x960 또는 640x480 사용
python pose_estimation_main.py test.mp4 --out_video_name ./test_result.mp4
```

폴더 구조는 다음과 같습니다.

```
├── ml  # 머신러닝 코드
    ├── face
        ├── models
            ├── cv_dnn
                ├── deploy.prototxt.txt
                ├── res10_300x300_ssd_iter_140000.caffemodel
            ├── dlib
                ├── shape_predictor_68_face_landmarks.dat
            ├── mmod_human_face_detector.dat
        └── detector.py
    └── io_event
```



* pip install -r requirements.txt 하면 필요 모듈 다운됩니다.
