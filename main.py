from fastapi import FastAPI
from MultiModal.dataset import MultimodalDataset
from MultiModal.model import CrossAttention, SoftLabelEncoder, ViTFeatureExtractor, ConditionClassifier
import torch
import torch.nn.functional as F
from warnings import filterwarnings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

filterwarnings(action='ignore')

app = FastAPI()
    
img_dim_h = 120  # 열화상 이미지 세로 크기
img_dim_w = 160  # 열화상 이미지 가로 크기
patch_size = 16
embed_dim = 128
num_heads = 4
depth = 6
aux_input_dim = 11  # 보조 데이터 차원 (예: 온도, 습도 등)
num_classes = 4  # 0:정상,1:관심,2:주의,3:위험

def load_AGV_model(Model_Parameter='Parameters/AGV_Best_State_Model.pth'):
    model = ConditionClassifier(img_dim_w, img_dim_h, patch_size, embed_dim, num_heads, depth, aux_input_dim, num_classes)
    model.load_state_dict(torch.load(Model_Parameter))
    return model.eval()

def load_OHT_model(Model_Parameter='Parameters/OHT_Best_State_Model.pth'):
    model = ConditionClassifier(img_dim_w, img_dim_h, patch_size, embed_dim, num_heads, depth, aux_input_dim, num_classes)
    model.load_state_dict(torch.load(Model_Parameter))
    return model.eval()

AGVConditionClassifier = load_AGV_model()
OHTConditionClassifier = load_OHT_model()