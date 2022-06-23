🏆 Dacon 코드 유사성 판단 경진대회
===
Dacon에서 실시한 코드 유사성 경진대회 코드이다.    
데이터셋 전처리 과정(cleaning 등)은 생략하였다.

### 💡 실행 방법

#### 1. Data Preprocessing
default로 모두 미리 설정해놨으며, `train_path`, `test_path`의 데이터셋 경로만 설정해주면 된다.
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_path  =TEST_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

#### 2. Training
```python
$ python train.py \
  --epochs =10
```

#### 3. Inference
```python
$ python inference.py\
  --loader_path   =TEST_DATALOADER_PATH            # test dataloader의 경로(data/test_dataloader.pkl)
  --save_sub_path =SAVE_SUBMISSION_FILE_PATH     # 저장할 submission.csv 파일 경로
```
