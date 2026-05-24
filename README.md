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
# 베이스라인만 (1차 커밋과 동일 설정)
python train.py --mode baseline

# 하이퍼파라미터 비교 실험만
python train.py --mode experiment

# 베이스라인 평가 + 실험 후 최고 F1 모델 저장 (기본)
python train.py --mode full
```

`--mode full` 실행 시 confusion matrix, classification report, 여러 `n_estimators`/`max_depth` 조합 비교표를 출력하고, test F1(macro)이 가장 높은 모델을 `models/wine_rf.pkl`에 저장합니다.

## 데이터셋

- 출처: [scikit-learn Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- 분류 문제 (회귀 아님)
