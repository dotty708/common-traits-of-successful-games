import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, classification_report

# -------------------------------------------------------------
# 1. 예시용 데이터 분할 (실제 X, y 데이터 대입)
# -------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------------------------------
# 2. 최빈값 기반 Dummy Classifier (가장 기초적인 베이스라인)
# -------------------------------------------------------------
# most_frequent: 무조건 가장 수가 많은 '군집 1'로만 예측
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
dummy_pred = dummy_model.predict(X_test)

dummy_acc = accuracy_score(y_test, dummy_pred)
# 다중 클래스이므로 recall_score에 average 옵션 지정 ('macro' 또는 'weighted')
dummy_recall_macro = recall_score(y_test, dummy_pred, average='macro')
dummy_recall_per_class = recall_score(y_test, dummy_pred, average=None)

print("=" * 40)
print("[베이스라인 1] 최빈값 예측 모델 (Dummy Classifier)")
print(f"• 정확도 (Accuracy)  : {dummy_acc:.4f}")
print(f"• 평균 재현율 (Macro Recall): {dummy_recall_macro:.4f}")
print(f"• 클래스별 재현율 (Recall)  : 군집0={dummy_recall_per_class[0]:.4f}, 군집1={dummy_recall_per_class[1]:.4f}, 군집2={dummy_recall_per_class[2]:.4f}")

# -------------------------------------------------------------
# 3. 로지스틱 회귀 (단순 머신러닝 베이스라인)
# -------------------------------------------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_recall_macro = recall_score(y_test, lr_pred, average='macro')
lr_recall_per_class = recall_score(y_test, lr_pred, average=None)

print("\n" + "=" * 40)
print("[베이스라인 2] 로지스틱 회귀 모델 (Logistic Regression)")
print(f"• 정확도 (Accuracy)  : {lr_acc:.4f}")
print(f"• 평균 재현율 (Macro Recall): {lr_recall_macro:.4f}")
print(f"• 클래스별 재현율 (Recall)  : 군집0={lr_recall_per_class[0]:.4f}, 군집1={lr_recall_per_class[1]:.4f}, 군집2={lr_recall_per_class[2]:.4f}")
print("=" * 40)

# 상세 정밀도/재현율/F1-score 리포트 출력
print("\n[Logistic Regression 상세 분류 리포트]")
print(classification_report(y_test, lr_pred, target_names=['군집 0', '군집 1', '군집 2']))
