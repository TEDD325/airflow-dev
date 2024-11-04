import pandas as pd

df = pd.read_csv("data/lung_cancer/survey_lung_cancer.csv")

print(df)
print(df.describe()) # 16개의 열 중 14개의 열이 숫자로 이루어진 것 같다.

print(f"예측 대상: {df.LUNG_CANCER}")