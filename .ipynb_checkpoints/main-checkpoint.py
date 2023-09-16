# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# No 'Access-Control-Allow-Origin'
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

import pickle

# /api_v1/mlmodelwithregression with dict params
# method : post
@app.post('/api_v1/mlmodelwithregression') 
def mlmodelwithregression(data:dict) : # json
    print('data with dict {}'.format(data))
    # data dict to 변수 활당
    Hospitalizationperiod = float(data['Hospitalizationperiod'])
    Painperiod = float(data['Painperiod'])
    age = float(data['age'])
    weight = float(data['weight'])
    LargeLymphocyte = float(data['LargeLymphocyte'])
    surgicaltechnique = float(data['surgicaltechnique'])

    encoded_data = float(data['encoded_data'])

    # pkl 파일 존재 확인 코드 필요

    # encoding 
    #result_encoding = 0;
    with open('dataset/RecurrenceOfSurgery_preprocessing_GB.pkl', 'rb') as regression_file:
        loaded_model = pickle.load(regression_file)
        surgicaltechnique = data['surgicaltechnique']
        # result_encoding = loaded_model.predict(input_labels)
        print('result_encoding: {}'.format(encoded_data))

    # # 예측값 리턴
    # result_encoding_ = {result_encoding[0,0]}
    # return result_encoding_

    # 학습 모델 불러와 예측
    with open('dataset/RecurrenceOfSurgery.pkl', 'rb') as regression_file:
        best_model = pickle.load(regression_file)
        input_labels = [[Hospitalizationperiod, Painperiod, age, weight, LargeLymphocyte, surgicaltechnique]] # 학습했던 설명변수 형식 맞게 적용
        result_predict = best_model.predict(input_labels)
        print('Predict Patient pain level Result : {}'.format(result_predict))
        pass

    # 예측값 리턴 # 결과를 하나의 딕셔너리에 저장하여 반환
    result = {'Patient_pain_level':result_predict[0]}
    return result

