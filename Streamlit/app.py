import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3

# Streamlit 앱 구성
st.title("Streamlit 대시보드")
st.write("센서 데이터 시각화")

# 데이터베이스 연결
conn = sqlite3.connect("sensor_data.db")

# 기기 선택택
df_option = st.selectbox("확인하고 싶은 기기를 입력하세요:", 
                         ["oht_12", "oht_13", "oht_14", "oht_15", "oht_16", "agv_12", "agv_13", "agv_14", "agv_15", "agv_16"])

# 센서 선택 메뉴
sensor_option = st.selectbox("확인하고 싶은 센서를 입력하세요:", ["NTC", "CT1", "CT2", "CT3", "CT4"])

# 선택된 기기별로 데이터 가져오기
query = f"SELECT * FROM {df_option}"
df = pd.read_sql(query, conn)

# 상태 레이블과 색상 정의
state_labels = {"0": "정상", "1": "관심", "2": "주의", "3": "위험"}
state_colors = {"정상": "green", "관심": "tan", "주의": "orange", "위험": "red"}

# 상태 레이블 추가
df["state_label"] = df["state"].map(state_labels)

# 사용자 선택 메뉴 (상태 선택)
state_option = st.selectbox("상태를 선택하세요:", ["전체", "정상", "관심", "주의", "위험"])

# 선택된 상태에 맞게 데이터 필터링
if state_option != "전체":
    filtered_df = df[df["state_label"] == state_option]
else:
    filtered_df = df

# 선택된 센서에 맞는 데이터 필터링
filtered_df = filtered_df[["state_label", sensor_option]]  # 선택된 센서만 추출

# 그래프 초기화
fig = go.Figure()

# 상태별 박스 플롯 추가 (필터링된 데이터 사용)
for state, color in state_colors.items():
    # 필터링된 데이터에서 특정 상태의 데이터만 추출
    state_data = filtered_df[filtered_df["state_label"] == state]
        
    # Box plot 추가
    fig.add_trace(
        go.Box(
            y=state_data[sensor_option],
            name=state,  # 상태 이름 (라벨)
            marker_color=color,
            boxmean="sd",  # 평균과 표준편차 표시
            hoverinfo="y",
        )
    )

# 그래프 레이아웃 설정
fig.update_layout(
    title={
        "text": f"State별 {sensor_option} 값 분포 - {df_option}",
        "x": 0.5,  # 중앙 정렬
        "xanchor": "center",
        "font": {"size": 20}
    },
    xaxis={
        "title": "State",
        "title_font": {"size": 16},
        "tickfont": {"size": 14},
    },
    yaxis={
        "title": f"{sensor_option} 값",
        "title_font": {"size": 16},
        "tickfont": {"size": 14},
        "gridcolor": "lightgrey",  # y축 격자선 색상
    },
    legend={
        "title": "상태별 분류",
        "font": {"size": 14},
    },
    plot_bgcolor="white",  # 배경색
    margin=dict(l=40, r=40, t=60, b=40),  # 여백 설정
)

# Streamlit에서 Plotly 차트 표시
st.plotly_chart(fig)
