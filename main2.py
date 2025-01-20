from MultiModal.dataset import MultimodalDataset, MultimodalTestDataset
from MultiModal.model import CrossAttention, SoftLabelEncoder, ViTFeatureExtractor, ConditionClassifier
from torch.utils.data import DataLoader
import joblib
import pandas as pd
import torch
import joblib
from Engine.utils import Evaluation_Classification_Model
import torch.nn.functional as F
from warnings import filterwarnings
filterwarnings(action='ignore')
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

device = 'cuda' # GPU 설정 | GPU가 없으면 'cpu'로 바꿔주세요.

img_dim_h = 120  # 열화상 이미지 세로 크기
img_dim_w = 160  # 열화상 이미지 가로 크기
patch_size = 16
embed_dim = 128
num_heads = 4
depth = 6
aux_input_dim = 11  # 보조 데이터 차원 (예: 온도, 습도 등)
num_classes = 4  # 0:정상,1:관심,2:주의,3:위험

# 2. MySQL 연결 정보 설정
user = 'root' # MySQL 사용자명
password = '0000' # Mysql 비밀번호
host = 'localhost' # MYSQL 서버 주소
database = 'sensor_data' # 데이터 베이스 이름

## 최적화 모델 불러오기
def load_AGV_model(Model_Parameter='Parameters/AGV_Best_State_Model.pth'):
    model = ConditionClassifier(img_dim_w, img_dim_h, patch_size, embed_dim, num_heads, depth, aux_input_dim, num_classes)
    model.load_state_dict(torch.load(Model_Parameter))
    return model.eval()

def load_OHT_model(Model_Parameter='Parameters/OHT_Best_State_Model.pth'):
    model = ConditionClassifier(img_dim_w, img_dim_h, patch_size, embed_dim, num_heads, depth, aux_input_dim, num_classes)
    model.load_state_dict(torch.load(Model_Parameter))
    return model.eval()

def connect_sql(user,password,host,database):
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")    
    return engine

def read_table_from_sql(engine,table_name):
        # AGV 테이블 데이터 읽기
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    return df

def DataPipeLine_From_SQL(user,password,host,database,batch_size=30):
    # MySQL데이터 베이스 연결
    engine = connect_sql(user,password,host,database)
    
    # 데이터 파이프라인 구축
    seperate_col = ['device_id','collection_date','collection_time','cumulative_operating_day']
    
    agv_dataset = read_table_from_sql(engine,'agv')
    agv_X= agv_dataset.drop(columns=seperate_col)

    oht_dataset = read_table_from_sql(engine,'oht')
    oht_X= oht_dataset.drop(columns=seperate_col)

    agv_test_dataset = MultimodalTestDataset(agv_X) # 파이토치 멀티모달 데이터셋 선언
    oht_test_dataset = MultimodalTestDataset(oht_X) # 파이토치 멀티모달 데이터셋 선언

    agv_test_dataloader = DataLoader(agv_test_dataset,batch_size=batch_size,shuffle=False) # agv_test_dataloader (데이터 파이프라인) 선언 # 배치 사이즈는 마음대로 정할 수 있습니다.
    oht_test_dataloader = DataLoader(oht_test_dataset,batch_size=batch_size,shuffle=False) # oht_test_dataloader (데이터 파이프라인) 선언
    return agv_test_dataloader, oht_test_dataloader

# 학습된 모델 불러오기
AGVConditionClassifier = load_AGV_model()
OHTConditionClassifier = load_OHT_model()

# MySQL데이터 베이스 연결
agv_test_dataloader,oht_test_dataloader = DataPipeLine_From_SQL(user,password,host,database,batch_size=600)
agv_test_dataloader = iter(agv_test_dataloader)

# 예측값
images, sensors = next(agv_test_dataloader)
print(torch.argmax(F.softmax(AGVConditionClassifier(images,sensors),dim=1),dim=1))