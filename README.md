# Wine Classification (OSS HW)

sklearn **Wine** 데이터셋(3클래스, 13특성)으로 품종을 분류하는 간단한 머신러닝 프로젝트입니다.  
수업에서 사용한 Iris, Digits 대신 다른 내장 데이터셋을 사용합니다.

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 학습 실행

```bash
python train.py
```

학습 후 `models/wine_rf.pkl`에 파이프라인(스케일러 + RandomForest)이 저장됩니다.

## 데이터셋

- 출처: [scikit-learn Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- 분류 문제 (회귀 아님)
