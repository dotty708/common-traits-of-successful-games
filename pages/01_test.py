import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb

# 1. 데이터 로드 (실제 데이터셋 경로 지정)
# df = pd.read_csv("game_cluster_data.csv")

# -------------------------------------------------------------
# [예시용 샘플 데이터생성 구조]
# Target: 군집 (0, 1, 2)
# Features: 게임 속성 (예: 장르, 플랫폼, 출시가격, 리뷰수, 개발기간 등)
# -------------------------------------------------------------

# X: 예측에 사용할 피처(게임 요소 데이터), y: 타겟 군집 (0, 1, 2)
# X = df.drop(columns=['cluster'])
# y = df['cluster']

# 2. Train / Test 데이터 분할 (클래스 비율 유지를 위해 stratify 사용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터 수: {len(X_train)}, 평가 데이터 수: {len(X_test)}")

# -------------------------------------------------------------
# 모델 1: Random Forest (기본 베이스라인)
# -------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("=== Random Forest 결과 ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred))

# -------------------------------------------------------------
# 모델 2: LightGBM (고성능 베이스라인)
# -------------------------------------------------------------
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    random_state=42,
    n_estimators=100
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

print("=== LightGBM 결과 ===")
print(f"Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")
print(classification_report(y_test, lgb_pred))

# -------------------------------------------------------------
# 3. 주요 영향 변수(Feature Importance) 확인
# -------------------------------------------------------------
import matplotlib.pyplot as plt

feature_importances = pd.Series(lgb_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (LightGBM)')
plt.xlabel('Importance Score')
plt.show()
