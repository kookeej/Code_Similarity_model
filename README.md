๐ Dacon ์ฝ๋ ์ ์ฌ์ฑ ํ๋จ ๊ฒฝ์ง๋ํ
===
Dacon์์ ์ค์ํ ์ฝ๋ ์ ์ฌ์ฑ ๊ฒฝ์ง๋ํ ์ฝ๋์๋๋ค. ๋ฐ์ดํฐ ํด๋ฆฌ๋ ๋ฐ augmentation๋ ์๋ตํ ์ฝ๋์๋๋ค.    

# 1. Preprocessing
* ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ๊ณผ์ ์์ ํฐ ๋ฐ์ดํฐ์ ํฌ๊ธฐ๋ก ์ธํ ๋ฉ๋ชจ๋ฆฌ ๋ถ์กฑ ์๋ฌ๋ฅผ ๋ฐฉ์งํ๊ธฐ ์ํด **chunck ๋จ์๋ก ํ ํฐํ**๋ฅผ ์งํํ์์ต๋๋ค.
```python
for i in tqdm(range(0, len(codes1), 256)):
    chunck1 = codes1[i:i+256]     # chunck ๋จ์๋ก ๋๋
    tokenized = tokenizer(
        chunck1,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_len
    )
```
* **๋ฏธ๋ฆฌ ๋ฐ์ดํฐ๋ก๋๋ฅผ `pickle`์ ์ด์ฉํด ์ง๋ ฌํํ์ฌ ์ ์ฅ**ํจ์ผ๋ก์จ ์คํ์ ์งํํ  ๋๋ง๋ค ๋ฐ์ดํฐ ๋ก๋ฉ ์์ ์์ด ๋ฐ๋ก ํ์ต, ์ถ๋ก ์ด ๊ฐ๋ฅํ๋๋ก ๋ง๋ค์์ต๋๋ค.
```python
# save
pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# loade
train_dataloader = pickle.load(open('data/train_dataloader.pkl', 'rb'))
```

# 2. Model
* `bi-encoder` ๊ตฌ์กฐ๋ฅผ ์ฌ์ฉํ์ฌ ์ค๊ณํ์ต๋๋ค.         
![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)


***


### ๐ก ์คํ ๋ฐฉ๋ฒ

#### 1. Data Preprocessing
default๋ก ๋ชจ๋ ๋ฏธ๋ฆฌ ์ค์ ํด๋จ์ผ๋ฉฐ, `train_path`, `test_path`์ ๋ฐ์ดํฐ์ ๊ฒฝ๋ก๋ง ์ค์ ํด์ฃผ๋ฉด ๋ฉ๋๋ค.
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
  --loader_path   =TEST_DATALOADER_PATH            # test dataloader์ ๊ฒฝ๋ก(data/test_dataloader.pkl)
  --save_sub_path =SAVE_SUBMISSION_FILE_PATH     # ์ ์ฅํ  submission.csv ํ์ผ ๊ฒฝ๋ก
```


***
# ๐ Results
**Public score**: 0.85885   
epoch: 6    

Ensemble, Regularzation, Data agmentation์ ํตํด ์ฑ๋ฅ ํฅ์์ ๋ธ๋ฆด ์ ์์
