import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from time import sleep
from tqdm import tqdm

import torch

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset

def inference(test_dataloader):
    # 예측값 저장 리스트
    preds_lst = []
    with torch.no_grad():
        print("Inference....")
        model.eval()
        bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for idx, items in bar:
            sleep(0.1)
            item = {key: val.to(device) for key,val in items.items()}
            outputs = model(**item)
            preds = torch.argmax(outputs, dim=-1)
            preds_lst.append(preds)
    
    return preds_lst
        

if __name__ == '__main__':
    config = DefaultConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader_path', type=str, default='data/test_dataloader.pkl', help="test dataloader path")
    parser.add_argument('--save_sub_path', type=str, default='data/{}.csv'.format(config.SAVE_SUB_FILE_NAME), help="submission file path")

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Test DataLoader...")
    test_dataloader = pickle.load(open(args.loader_path, 'rb'))
    
    print("Loading saved model...")
    model = CustomModel(config.MODEL_CONFIG)
    model.to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    
    # Inference
    preds_lst = inference(test_dataloader)
    print("Inference Finish!")
    
    # 최종 예측 리스트 만들기
    final_preds = sum([preds_lst[i].tolist() for i in range(len(preds_lst))], [])
    print("Final Prediction Length:", len(final_preds))
    
    # submission 저장
    print("Save final submission...")
    sub = pd.read_csv("data/sample_submission.csv")
    sub['similar'] = final_preds
    sub.to_csv(args.save_sub_path, index=False)
    
    print("Complete!")