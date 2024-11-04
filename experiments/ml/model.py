import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from generate_data import generate_similar_data

# 1. 데이터 로드 및 기본 전처리
def load_and_preprocess_data(data):
    # 컬럼명 소문자로 변경
    data.columns = data.columns.str.lower()
    
    # LUNG_CANCER 타겟 변수를 이진값으로 변환
    le = LabelEncoder()
    data['lung_cancer'] = le.fit_transform(data['lung_cancer'])
    
    # gender 변수 이진값으로 변환
    data['gender'] = le.fit_transform(data['gender'])
    
    return data

# 2. 특성 중요도 시각화 함수
def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.show()

# 3. 메인 모델링 파이프라인
def train_catboost_model(data):
    # 데이터 전처리
    data = load_and_preprocess_data(data)
    
    # 특성과 타겟 분리
    X = data.drop(['lung_cancer'], axis=1)
    y = data['lung_cancer']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # CatBoost 모델 정의
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        random_seed=42,
        verbose=100
    )
    
    # 교차 검증 수행
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n교차 검증 점수: {cv_scores}")
    print(f"평균 교차 검증 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 전체 학습 데이터로 최종 모델 학습
    model.fit(X_train, y_train)
    
    # 테스트 세트로 예측
    y_pred = model.predict(X_test)
    
    # 모델 성능 평가
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 특성 중요도 시각화
    plot_feature_importance(model, X.columns)
    
    return model

# 데이터 로드 및 모델 학습 실행
data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')
model = train_catboost_model(data)

# 새로운 데이터에 대한 예측 함수
def predict_lung_cancer(model, new_data):
    """
    새로운 데이터에 대한 폐암 위험도를 예측합니다.
    
    Parameters:
    - model: 학습된 CatBoost 모델
    - new_data: 예측하고자 하는 새로운 데이터 (DataFrame)
    
    Returns:
    - 예측 결과 (0: 폐암 위험 낮음, 1: 폐암 위험 높음)
    - 예측 확률
    """
    new_data = load_and_preprocess_data(new_data.copy())
    if 'lung_cancer' in new_data.columns:
        new_data = new_data.drop(['lung_cancer'], axis=1)
    
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[:, 1]
    
    return prediction, probability


# 원본 데이터 로드
data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')

# 새로운 유사 데이터 10개 생성
new_samples = generate_similar_data(data, n_samples=10)

# 결과 출력
print("\n생성된 새로운 데이터 샘플:")
print(new_samples.to_string(index=False))
    
pred, prob = predict_lung_cancer(model, new_samples)
print(f"prediced value: {pred}")
print(f"probability value: {prob}")