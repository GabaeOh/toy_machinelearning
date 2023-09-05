#!/usr/bin/env python
# coding: utf-8

# ## 분석 목적
# - 서비스 제공 대상 : 제약 회사 
# - 분석 목적 : 환자통증정도와 관련된 입원/통증 기간 분석, 환자 특성 변화 분석 등
#     - 입원/통증 기간 분석
#         - '입원기간'과 '통증기간(월)' 등의 데이터를 활용하여 환자들의 치료 /통증기간과 효과를 분석하고자 한다.
#         -  제품이 어떤 기간 동안 환자의 상태 개선에 도움이 되는지를 확인하여 제품 효과를 검토할 수 있다.
#     - 환자 특성 변화 분석
#         - 환자들의 연령, 성별, 체중 등 특성을 파악하여 제약회사의 대상 구매고객의 프로필과 대조 분석한다(추후)
#         - 이를 통해 어떤 연령층이나 성별에 해당하는 환자들에게 제품 또는 서비스를 맞춤화하여 제공할 수 있는지 확인하고자 한다.

# ## 분석 결과
# - 정확도(accuracy) :0.988  --> 높은 정확도로, 모델이 예측을 매우 잘 수행함
# - precision, recall, f1-score : 0.99
#     - 높은 정밀도, 재현율, F1 점수를 갖고 있음 -->  이 모델은 양성과 음성 클래스 모두를 높은 성능으로 예측하고 있음

# In[1]:


import pandas as pd


# In[2]:


df_ROS = pd.read_csv('../dataset/RecurrenceOfSurgery.csv')
df_ROS[:2]


# In[3]:


df_ROS.columns


# In[4]:


df_ROS['환자통증정도'].value_counts()


# ### 변수 선택 
# - 목표변수(target) : 환자통증정도
# - 설명변수(features) : '입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법'

# In[5]:


df_ROS_extract = df_ROS[['환자통증정도','입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법']]


# In[6]:


df_ROS_extract.info() # 설명변수 중 범주형 : 수술기법


# ### Pre-Processing (전처리과정)

# #### Imputation (결측값 처리)

# In[7]:


df_ROS_extract.isnull().sum()


# #### 최빈값으로 결측치 처리 
# - 결측치 처리하고자 하는 열 : 수술기법

# In[8]:


df_ROS_extract['수술기법'].value_counts()


# In[9]:


df_ROS_extract['수술기법'].mode() #최빈값 확인


# In[10]:


# 최빈값으로 결측치 채우기
df_ROS_extract['수술기법'] = df_ROS_extract['수술기법'].fillna(df_ROS_extract['수술기법'].mode().iloc[0]) 
## .iloc[0] --> 첫번째 행 / 최빈값이 2개 이상일경우 첫번째 행의 최빈값 값을 의미


# In[12]:


df_ROS_extract['수술기법'].isnull().sum()


# In[13]:


df_ROS_extract.isnull().sum()


# In[14]:


df_ROS_dropna = df_ROS_extract.dropna() # null값 제거한 df
df_ROS_dropna.isnull().sum()


# #### Encoding with One Hot Encoding
# 범주형 데이터를 연속형 숫자 데이터로 변환하고자 하는 열 : 수술기법

# In[15]:


df_ROS_dropna['수술기법'].value_counts()


# In[16]:


from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder() #인스턴스화 
oneHotEncoder.fit(df_ROS_dropna[['수술기법']])# 학습
oneHotEncoder.categories_


# In[17]:


encoded_data = oneHotEncoder.transform(df_ROS_dropna[['수술기법']]).toarray()
encoded_data


# In[18]:


df_encoded_data = pd.DataFrame(data=encoded_data, columns=oneHotEncoder.get_feature_names_out(['수술기법']))
df_encoded_data[:3]


# #### concat(병합)

# In[19]:


df_ROS_concat = pd.concat([df_ROS_dropna.reset_index(drop=True), df_encoded_data.reset_index(drop=True)], axis=1 )
df_ROS_concat


# ### 예측모델을 통한 결측값 채우기
# - 채우고자 하는 열 : 통증기간(월)

# In[59]:


df_ROS_dropna.columns


# In[20]:


target_forfillingna = df_ROS_dropna['통증기간(월)']
features_forfillingna = df_ROS_dropna.drop(columns=['통증기간(월)','수술기법'])


# In[21]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=False) #모델 초기화 --> fit_intercept=False
model.fit(features_forfillingna, target_forfillingna) #모델 훈련


# In[22]:


def predict_missing_duration(row):
    if pd.isnull(row['통증기간(월)']):
        # 모델을 사용하여 결측치 예측 
        #  reshape(1, -1) --> array값으로 나오기때문에 dataframe으로 바꿔주고 컬럼을가져옴
        features_forfillingna = row[['환자통증정도', '입원기간', '연령', '체중', 'Large Lymphocyte']].values.reshape(1, -1)
        predicted_duration = model.predict(features_forfillingna)
        return predicted_duration[0]
    else:
        # 값이 있는 경우 그대로 반환
        return row['통증기간(월)']


# In[23]:


df_ROS_concat['통증기간(월)'] = df_ROS_extract.apply(predict_missing_duration, axis=1)


# In[24]:


df_ROS_concat['통증기간(월)'].isnull().sum()


# In[25]:


df_ROS_concat.isnull().sum()


# #### Scaling - MinMaxScaler

# In[60]:


df_ROS_concat.columns


# In[26]:


target = df_ROS_concat['환자통증정도']
features = df_ROS_concat.drop(columns=['환자통증정도', '수술기법'] )


# In[27]:


features.columns


# In[28]:


from sklearn.preprocessing import MinMaxScaler
minMaxScaler= MinMaxScaler()
features = minMaxScaler.fit_transform(features)


# #### Imbalanced Data Sampling
# -  Under Sampling : Tomek's Link 

# In[29]:


from imblearn.under_sampling import TomekLinks
from sklearn.datasets import make_classification


# In[30]:


features, target = make_classification(n_classes=2, class_sep=2,
                    weights=[0.3, 0.7], n_informative=3, n_redundant=1, flip_y=0,
                    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)


# In[31]:


features.shape, target.shape


# In[32]:


from collections import Counter
Counter(target)


# In[33]:


tomekLinks = TomekLinks() #인스턴스화
features_resaple, target_resample = tomekLinks.fit_resample(features, target) #학습


# In[34]:


features_resaple.shape, target_resample.shape


# In[35]:


Counter(target_resample)


# ### 정형화

# #### split

# In[36]:


from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=111)
features_train.shape, target_train.shape, features_test.shape, target_test.shape


# ### 모델 학습
# - 목표변수가 범주형일 경우 : DecisionTreeClassifier
# - 목표변수가 연속형일 경우 : DecisionTreeRegressor

# In[37]:


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
model = DecisionTreeClassifier() 
from sklearn.model_selection import GridSearchCV


# In[47]:


hyper_params = {'min_samples_leaf' : range(2,6) #[2,3,4,5,6]
               ,'max_depth' :  range(2,6)
               ,'min_samples_split' :  range(2,6)}


# In[51]:


#from sklearn.metrics import f1_score, make_scorer
grid_search = GridSearchCV(model, param_grid=hyper_params, cv=3, verbose=1) # scoring=scoring


# In[52]:


grid_search.fit(features_train, target_train)


# In[53]:


grid_search.best_estimator_


# In[54]:


grid_search.best_score_, grid_search.best_params_ 

## best_score : 0.988 /정확도(accuracy) --> 높은 정확도로, 모델이 예측을 매우 잘 수행함


# In[55]:


best_model = grid_search.best_estimator_ # 하나의 모델 --> 그 중에서 최고의 모델
best_model 


# In[56]:


target_test_predict = best_model.predict(features_test)
target_test_predict


# ### 평가

# In[61]:


from sklearn.metrics import classification_report
print(classification_report(target_test, target_test_predict))

# support는 각 클래스에 속하는 샘플의 개수 -->  클래스 0은 76개의 샘플이 있고, 클래스 1은 174개의 샘플이 있음


# In[ ]:




