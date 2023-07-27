# Parkinson Project (2023.04.24 ~ 2023.05.17)

[대시보드 링크](https://kimmeatan-kaggle-p-app-pjrdqh.streamlit.app/) <br/><br/>
[포트폴리오](pdf/2조_파킨슨_질병_진단예측.pdf)<br/><br/>
![screensh](img/홈페이지.png)

## 목적
파킨슨병 환자의 임상 데이터를 사용하여, 파킨슨병 환자의 진행을 측정하는 MDS-UPDRS(통합 파킨슨 평가청도) 예측

## 데이터
캐글(Kaggle)의 AMP - Parkinson's Disease Progression Prediction 주제의 데이터를 사용
(train_clinical_data.csv, supplemental_clinical_data.csv, train_peptides.csv, train_protein.csv)

## ERD
![screensh](img/erd.png)

## 팀 구성

- 사용언어 : Python 3.9.13v
- 작업 툴 : google colab, pycharm, kaggle
- 인원 : 5명
- 주요 업무 : Streamlit 라이브러리를 이용한 웹개발 구현 코드 작성 및 머신러닝을 활용한 파킨슨 질병 진단 예측
- 기간 : 2023-04-27 ~ 2023-05-16 
- 
## 주요 기능

- ***홈페이지(Home)***
  + 대회목표 및 대회 개요
  
- ***설명(Description)***
  + 파킨슨병 / MDS-UPDRS 점수 설명
  + 평가지표 설명
  + 대회에 사용되는 용어 설명
  
- ***데이터(Data)***
  + 데이터셋 컬럼 확인
  + 데이터셋 확인
  
- ***탐색적자료분석(EDA)***
  + UPDRS 점수와 약물복용여부 그래프
  + Null값 분석 및 비교
  + clinical 데이터, supplemental 데이터 분포 확인
  + protein 데이터, peptide 데이터 분포 확인
  + protein 변동계수에 따른 상관관계
  + peptide 변동계수에 다른 상관관계 
  
- ***통계(Stat)*** 
  + 기초통계분석
  + 상관분석
  + 평가지표
  
- ***모델링(ML)***
  + Catboost 설명
  + 교차검증
  + 모델 구축
  + 모델 성능 평가

## 📥 설치 방법
### Windows
- 버전 확인
  - Python : 3.10.10
  - 라이브러리 : matplotlib(3.7.1), numpy(1.24.3), pandas(2.0.1), pingouin(0.5.3), plotly(5.14.1), plotly-express(0.4.1), scikit-learn(1.2.2), seaborn(0.12.2), statsmodels(0.13.5), streamlit(1.22.0), streamlit-option-menu(0.3.2), utils(1.0.1), xgboost(1.7.5)

- 프로젝트 파일을 다운로드 받습니다.
```
git clone https://github.com/kimmeatan/storesales_streamlit_by_Kaggle.git
```
- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 10 기준)
```
virtualenv venv
source venv/Scripts/activate
```
- 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
- streamlit 명령어를 실행합니다.
```
streamlit run app.py
```
