import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
def train_step(model:torch.nn.Module,
               train_loader:torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim,
               device:torch.device='cuda'):
    model.train()
    # 모델도 GPU 환경으로 보내기 (모델 학습시 GPU 학습 가속화)
    model = model.to(device)
    train_loss, train_acc = 0.0, 0.0
    for batch_images,batch_features,target in tqdm(train_loader):
        # GPU 상으로 데이터 보내기 (모델 학습시 GPU 학습 가속화)
        batch_images,batch_features,target = batch_images.to(device),batch_features.to(device),target.to(device)
        # 데이터 -> 모델 입력 
        y_pred = model(batch_images,batch_features)
        # 로스값 계산
        loss = loss_fn(y_pred,target)
        train_loss += loss.item()
        # 최적화 기울기 초기화
        optimizer.zero_grad()
        # 역전파 Loss Backward
        loss.backward()
        # Optimizer Step 가중치 업데이트
        optimizer.step()
        # 정확도 계산
        y_pred_class = torch.argmax(F.softmax(y_pred,dim=1),dim=1)
        train_acc += ((y_pred_class==target).sum().item()/len(y_pred))
    train_loss,train_acc = train_loss/len(train_loader), train_acc/len(train_loader)
    return train_loss,train_acc

def test_step(model:torch.nn.Module,
              test_loader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device='cuda'):
    # 평가 모드
    model.eval()
    test_loss,test_acc = 0.0, 0.0
    # 추론 모드
    with torch.inference_mode():
        for batch_images,batch_features,target in tqdm(test_loader):
            # GPU 상으로 데이터 보내기 (모델 학습시 GPU 학습 가속화)
            batch_images,batch_features,target = batch_images.to(device),batch_features.to(device),target.to(device)
            
            # 데이터 -> 모델 입력 
            test_pred_logits = model(batch_images,batch_features)
            
            # 로스 계산
            loss = loss_fn(test_pred_logits,target)
            test_loss += loss.item()
            
            # 정확도 계산
            test_pred_class = torch.argmax(F.softmax(test_pred_logits,dim=1),dim=1)
            test_acc += ((test_pred_class==target).sum().item()/len(test_pred_logits))
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)        
    return test_loss,test_acc

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()