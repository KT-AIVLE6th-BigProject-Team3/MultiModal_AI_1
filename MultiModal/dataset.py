import torch
import pandas
import numpy as np
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """사용자 정의 멀티모달 데이터셋"""
    def __init__(self,X:pandas.DataFrame,y:pandas.Series):
        """필요한 데이터를 이곳에서 선언
        Parameter:
        X: 열화상 이미지와 센서데이터의 데이터 프레임
        y: 정답 데이터의
        """
        self.X = X
        self.y = y
    def __getitem__(self, index):
        """열화상 이미지 [1,120,160]크기와 센서 데이터 11개의 칼럼 그리고 이에 해당하는 정답 데이터 반환"""
        image = torch.tensor(np.load(self.X.iloc[index]['filenames']),dtype=torch.float32).unsqueeze(0)
        sensor_features = self.X.drop(columns=['filenames'])
        sensor_features = torch.tensor(sensor_features.iloc[index].values,dtype=torch.float32)
        label = int(self.y.iloc[index]) # 나중에 학습하는데 있어서 느려지게하는 요인이면 데이터 프레임 자체에서 변경함함
        return image, sensor_features, label
    
    def __len__(self):
        return len(self.X)
    
class MultimodalTestDataset(Dataset):
    """사용자 정의 멀티모달 데이터셋"""
    def __init__(self,X:pandas.DataFrame):
        """필요한 데이터를 이곳에서 선언
        Parameter:
        X: 열화상 이미지와 센서데이터의 데이터 프레임
        """
        self.X = X
    def __getitem__(self, index):
        """열화상 이미지 [1,120,160]크기와 센서 데이터 11개의 칼럼 그리고 이에 해당하는 정답 데이터 반환"""
        image = torch.tensor(np.load(self.X.iloc[index]['filenames']),dtype=torch.float32).unsqueeze(0)
        sensor_features = self.X.drop(columns=['filenames'])
        sensor_features = torch.tensor(sensor_features.iloc[index].values,dtype=torch.float32)
        return image, sensor_features
    
    def __len__(self):
        return len(self.X)    