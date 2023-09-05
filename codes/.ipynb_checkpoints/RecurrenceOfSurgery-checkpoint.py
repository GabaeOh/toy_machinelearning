{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8fb28df7-3a5d-466f-bb16-e8734ac8253b",
   "metadata": {},
   "source": [
    "## 분석 목적\n",
    "- 서비스 제공 대상 : 제약 회사 \n",
    "- 분석 목적 : 환자통증정도와 관련된 입원/통증 기간 분석, 환자 특성 변화 분석 등\n",
    "    - 입원/통증 기간 분석\n",
    "        - '입원기간'과 '통증기간(월)' 등의 데이터를 활용하여 환자들의 치료 /통증기간과 효과를 분석하고자 한다.\n",
    "        -  제품이 어떤 기간 동안 환자의 상태 개선에 도움이 되는지를 확인하여 제품 효과를 검토할 수 있다.\n",
    "    - 환자 특성 변화 분석\n",
    "        - 환자들의 연령, 성별, 체중 등 특성을 파악하여 제약회사의 대상 구매고객의 프로필과 대조 분석한다(추후)\n",
    "        - 이를 통해 어떤 연령층이나 성별에 해당하는 환자들에게 제품 또는 서비스를 맞춤화하여 제공할 수 있는지 확인하고자 한다."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "46e4d334-2111-4b30-85a8-1cef965be5a0",
   "metadata": {},
   "source": [
    "## 분석 결과\n",
    "- 정확도(accuracy) :0.988  --> 높은 정확도로, 모델이 예측을 매우 잘 수행함\n",
    "- precision, recall, f1-score : 0.99\n",
    "    - 높은 정밀도, 재현율, F1 점수를 갖고 있음 -->  이 모델은 양성과 음성 클래스 모두를 높은 성능으로 예측하고 있음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "99c2b420-7757-46c7-a9a7-e3ba13a45307",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e647d9aa-bcd5-4574-bc1e-ed187756bd7c",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>환자ID</th>\n",
       "      <th>Large Lymphocyte</th>\n",
       "      <th>Location of herniation</th>\n",
       "      <th>ODI</th>\n",
       "      <th>가족력</th>\n",
       "      <th>간질성폐질환</th>\n",
       "      <th>고혈압여부</th>\n",
       "      <th>과거수술횟수</th>\n",
       "      <th>당뇨여부</th>\n",
       "      <th>...</th>\n",
       "      <th>Modic change</th>\n",
       "      <th>PI</th>\n",
       "      <th>PT</th>\n",
       "      <th>Seg Angle(raw)</th>\n",
       "      <th>Vaccum disc</th>\n",
       "      <th>골밀도</th>\n",
       "      <th>디스크단면적</th>\n",
       "      <th>디스크위치</th>\n",
       "      <th>척추이동척도</th>\n",
       "      <th>척추전방위증</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1PT</td>\n",
       "      <td>22.8</td>\n",
       "      <td>3</td>\n",
       "      <td>51.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>51.6</td>\n",
       "      <td>36.6</td>\n",
       "      <td>14.4</td>\n",
       "      <td>0</td>\n",
       "      <td>-1.01</td>\n",
       "      <td>2048.5</td>\n",
       "      <td>4</td>\n",
       "      <td>Down</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2PT</td>\n",
       "      <td>44.9</td>\n",
       "      <td>4</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>40.8</td>\n",
       "      <td>7.2</td>\n",
       "      <td>17.8</td>\n",
       "      <td>0</td>\n",
       "      <td>-1.14</td>\n",
       "      <td>1753.1</td>\n",
       "      <td>4</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 52 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0 환자ID  Large Lymphocyte  Location of herniation   ODI  가족력  \\\n",
       "0           0  1PT              22.8                       3  51.0  0.0   \n",
       "1           1  2PT              44.9                       4  26.0  0.0   \n",
       "\n",
       "   간질성폐질환  고혈압여부  과거수술횟수  당뇨여부  ...  Modic change    PI    PT  Seg Angle(raw)  \\\n",
       "0       0      0       0     0  ...             3  51.6  36.6            14.4   \n",
       "1       0      0       0     0  ...             0  40.8   7.2            17.8   \n",
       "\n",
       "   Vaccum disc   골밀도  디스크단면적  디스크위치  척추이동척도  척추전방위증  \n",
       "0            0 -1.01  2048.5      4    Down       0  \n",
       "1            0 -1.14  1753.1      4      Up       0  \n",
       "\n",
       "[2 rows x 52 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS = pd.read_csv('../dataset/RecurrenceOfSurgery.csv')\n",
    "df_ROS[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a40d106f-cdc0-45c6-bf1a-448d44f0ae62",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Unnamed: 0', '환자ID', 'Large Lymphocyte', 'Location of herniation',\n",
       "       'ODI', '가족력', '간질성폐질환', '고혈압여부', '과거수술횟수', '당뇨여부', '말초동맥질환여부', '빈혈여부',\n",
       "       '성별', '스테로이드치료', '신부전여부', '신장', '심혈관질환', '암발병여부', '연령', '우울증여부', '입원기간',\n",
       "       '입원일자', '종양진행여부', '직업', '체중', '퇴원일자', '헤모글로빈수치', '혈전합병증여부', '환자통증정도',\n",
       "       '흡연여부', '통증기간(월)', '수술기법', '수술시간', '수술실패여부', '수술일자', '재발여부', '혈액형',\n",
       "       '전방디스크높이(mm)', '후방디스크높이(mm)', '지방축적도', 'Instability', 'MF + ES',\n",
       "       'Modic change', 'PI', 'PT', 'Seg Angle(raw)', 'Vaccum disc', '골밀도',\n",
       "       '디스크단면적', '디스크위치', '척추이동척도', '척추전방위증'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b53569cb-e365-4dd4-8e62-3c2240b4d9bb",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7     885\n",
       "8     410\n",
       "9     172\n",
       "2     130\n",
       "10    107\n",
       "1      52\n",
       "5      46\n",
       "3      44\n",
       "6      40\n",
       "4       8\n",
       "Name: 환자통증정도, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS['환자통증정도'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a0d946a-320f-443b-ae4f-3946aa6811ef",
   "metadata": {},
   "source": [
    "### 변수 선택 \n",
    "- 목표변수(target) : 환자통증정도\n",
    "- 설명변수(features) : '입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e69ecb1b-cbf9-4b08-b7e9-20b3163e3082",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_ROS_extract = df_ROS[['환자통증정도','입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "04d6fe82-991f-4cd8-a493-c1c218284c7c",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1894 entries, 0 to 1893\n",
      "Data columns (total 7 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   환자통증정도            1894 non-null   int64  \n",
      " 1   입원기간              1894 non-null   int64  \n",
      " 2   통증기간(월)           1890 non-null   float64\n",
      " 3   연령                1894 non-null   int64  \n",
      " 4   체중                1894 non-null   float64\n",
      " 5   Large Lymphocyte  1894 non-null   float64\n",
      " 6   수술기법              1813 non-null   object \n",
      "dtypes: float64(3), int64(3), object(1)\n",
      "memory usage: 103.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df_ROS_extract.info() # 설명변수 중 범주형 : 수술기법"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5d4a3f6-1fae-48aa-8a2a-5c71ac337d31",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Pre-Processing (전처리과정)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00ff0e25-8c24-406a-ad2c-2107de273637",
   "metadata": {},
   "source": [
    "#### Imputation (결측값 처리)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ccc41c06-e347-4161-9f2f-4f0348bed978",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "환자통증정도               0\n",
       "입원기간                 0\n",
       "통증기간(월)              4\n",
       "연령                   0\n",
       "체중                   0\n",
       "Large Lymphocyte     0\n",
       "수술기법                81\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_extract.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04ad474a-33e6-4682-b2d1-9e1e3807fed9",
   "metadata": {},
   "source": [
    "#### 최빈값으로 결측치 처리 \n",
    "- 결측치 처리하고자 하는 열 : 수술기법"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7f768a07-b310-42d2-b953-39f9a1251e19",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TELD    1673\n",
       "IELD     140\n",
       "Name: 수술기법, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_extract['수술기법'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2046bdab-9ad2-4c5b-99c9-21c7b6e23fdc",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    TELD\n",
       "Name: 수술기법, dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_extract['수술기법'].mode() #최빈값 확인"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "05b38e5c-509b-4f43-b36d-e8156c205f2b",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\02-19\\AppData\\Local\\Temp\\ipykernel_7524\\2552403456.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_ROS_extract['수술기법'] = df_ROS_extract['수술기법'].fillna(df_ROS_extract['수술기법'].mode().iloc[0])\n"
     ]
    }
   ],
   "source": [
    "# 최빈값으로 결측치 채우기\n",
    "df_ROS_extract['수술기법'] = df_ROS_extract['수술기법'].fillna(df_ROS_extract['수술기법'].mode().iloc[0]) \n",
    "## .iloc[0] --> 첫번째 행 / 최빈값이 2개 이상일경우 첫번째 행의 최빈값 값을 의미"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8beea019-ac48-469b-94a8-a07df83ea486",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_extract['수술기법'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f91fc32a-19e6-4a75-9fd4-74bdb2650d1c",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "환자통증정도              0\n",
       "입원기간                0\n",
       "통증기간(월)             4\n",
       "연령                  0\n",
       "체중                  0\n",
       "Large Lymphocyte    0\n",
       "수술기법                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_extract.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ce025598-93c8-470f-be54-7853b00da9fe",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "환자통증정도              0\n",
       "입원기간                0\n",
       "통증기간(월)             0\n",
       "연령                  0\n",
       "체중                  0\n",
       "Large Lymphocyte    0\n",
       "수술기법                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_dropna = df_ROS_extract.dropna() # null값 제거한 df\n",
    "df_ROS_dropna.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "863306ee-b962-4c7f-a947-5459cacde0c2",
   "metadata": {},
   "source": [
    "#### Encoding with One Hot Encoding\n",
    "범주형 데이터를 연속형 숫자 데이터로 변환하고자 하는 열 : 수술기법"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ef6e7b04-6bd3-4977-90c3-6c46c3a0fe24",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TELD    1750\n",
       "IELD     140\n",
       "Name: 수술기법, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_dropna['수술기법'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "5117cb0e-4a79-4b95-a8fc-ebdfa565b3ca",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array(['IELD', 'TELD'], dtype=object)]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder\n",
    "oneHotEncoder = OneHotEncoder() #인스턴스화 \n",
    "oneHotEncoder.fit(df_ROS_dropna[['수술기법']])# 학습\n",
    "oneHotEncoder.categories_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ed5225e9-242f-40b5-9422-ba3d189fec66",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 1.],\n",
       "       [0., 1.],\n",
       "       [0., 1.],\n",
       "       ...,\n",
       "       [1., 0.],\n",
       "       [0., 1.],\n",
       "       [0., 1.]])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "encoded_data = oneHotEncoder.transform(df_ROS_dropna[['수술기법']]).toarray()\n",
    "encoded_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "cbdc0cd2-4a84-43e6-aa42-24eba2cc19ce",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>수술기법_IELD</th>\n",
       "      <th>수술기법_TELD</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   수술기법_IELD  수술기법_TELD\n",
       "0        0.0        1.0\n",
       "1        0.0        1.0\n",
       "2        0.0        1.0"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_encoded_data = pd.DataFrame(data=encoded_data, columns=oneHotEncoder.get_feature_names_out(['수술기법']))\n",
    "df_encoded_data[:3]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0645ef2-d80a-4bc5-ade0-4a8202f9ebde",
   "metadata": {},
   "source": [
    "#### concat(병합)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ef2cacab-6b3e-494d-95f4-934cf5fb8d2f",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>환자통증정도</th>\n",
       "      <th>입원기간</th>\n",
       "      <th>통증기간(월)</th>\n",
       "      <th>연령</th>\n",
       "      <th>체중</th>\n",
       "      <th>Large Lymphocyte</th>\n",
       "      <th>수술기법</th>\n",
       "      <th>수술기법_IELD</th>\n",
       "      <th>수술기법_TELD</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>66</td>\n",
       "      <td>60.3</td>\n",
       "      <td>22.8</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>47</td>\n",
       "      <td>71.7</td>\n",
       "      <td>44.9</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>39</td>\n",
       "      <td>77.1</td>\n",
       "      <td>53.0</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>40</td>\n",
       "      <td>74.2</td>\n",
       "      <td>54.7</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>42</td>\n",
       "      <td>80.7</td>\n",
       "      <td>53.8</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1885</th>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>12.0</td>\n",
       "      <td>59</td>\n",
       "      <td>64.0</td>\n",
       "      <td>44.0</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1886</th>\n",
       "      <td>7</td>\n",
       "      <td>4</td>\n",
       "      <td>6.0</td>\n",
       "      <td>42</td>\n",
       "      <td>59.0</td>\n",
       "      <td>30.8</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1887</th>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>61</td>\n",
       "      <td>70.0</td>\n",
       "      <td>36.7</td>\n",
       "      <td>IELD</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1888</th>\n",
       "      <td>7</td>\n",
       "      <td>4</td>\n",
       "      <td>24.0</td>\n",
       "      <td>29</td>\n",
       "      <td>77.0</td>\n",
       "      <td>32.5</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1889</th>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>6.0</td>\n",
       "      <td>30</td>\n",
       "      <td>49.0</td>\n",
       "      <td>49.5</td>\n",
       "      <td>TELD</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1890 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      환자통증정도  입원기간  통증기간(월)  연령    체중  Large Lymphocyte  수술기법  수술기법_IELD  \\\n",
       "0         10     2      1.0  66  60.3              22.8  TELD        0.0   \n",
       "1         10     1      1.0  47  71.7              44.9  TELD        0.0   \n",
       "2          7     1      1.0  39  77.1              53.0  TELD        0.0   \n",
       "3          7     1      2.0  40  74.2              54.7  TELD        0.0   \n",
       "4          7     2      1.0  42  80.7              53.8  TELD        0.0   \n",
       "...      ...   ...      ...  ..   ...               ...   ...        ...   \n",
       "1885       7     2     12.0  59  64.0              44.0  TELD        0.0   \n",
       "1886       7     4      6.0  42  59.0              30.8  TELD        0.0   \n",
       "1887       7     3      1.0  61  70.0              36.7  IELD        1.0   \n",
       "1888       7     4     24.0  29  77.0              32.5  TELD        0.0   \n",
       "1889       8     2      6.0  30  49.0              49.5  TELD        0.0   \n",
       "\n",
       "      수술기법_TELD  \n",
       "0           1.0  \n",
       "1           1.0  \n",
       "2           1.0  \n",
       "3           1.0  \n",
       "4           1.0  \n",
       "...         ...  \n",
       "1885        1.0  \n",
       "1886        1.0  \n",
       "1887        0.0  \n",
       "1888        1.0  \n",
       "1889        1.0  \n",
       "\n",
       "[1890 rows x 9 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_concat = pd.concat([df_ROS_dropna.reset_index(drop=True), df_encoded_data.reset_index(drop=True)], axis=1 )\n",
    "df_ROS_concat"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d8d7279-2b86-4f1d-b266-919a69d08aa0",
   "metadata": {},
   "source": [
    "### 예측모델을 통한 결측값 채우기\n",
    "- 채우고자 하는 열 : 통증기간(월)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "401f5f3c-828a-49a6-bab9-f151fdeb8589",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['환자통증정도', '입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법'], dtype='object')"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_dropna.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "73b06951-cf8c-44a9-8237-7c8bb217cb97",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "target_forfillingna = df_ROS_dropna['통증기간(월)']\n",
    "features_forfillingna = df_ROS_dropna.drop(columns=['통증기간(월)','수술기법'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "228be7c0-c29f-4987-b4f6-a51f8ef58d53",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression(fit_intercept=False)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression(fit_intercept=False)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression(fit_intercept=False)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "model = LinearRegression(fit_intercept=False) #모델 초기화 --> fit_intercept=False\n",
    "model.fit(features_forfillingna, target_forfillingna) #모델 훈련"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6064280b-1ecc-41d9-9ca0-9760b6c4ca2a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def predict_missing_duration(row):\n",
    "    if pd.isnull(row['통증기간(월)']):\n",
    "        # 모델을 사용하여 결측치 예측 \n",
    "        #  reshape(1, -1) --> array값으로 나오기때문에 dataframe으로 바꿔주고 컬럼을가져옴\n",
    "        features_forfillingna = row[['환자통증정도', '입원기간', '연령', '체중', 'Large Lymphocyte']].values.reshape(1, -1)\n",
    "        predicted_duration = model.predict(features_forfillingna)\n",
    "        return predicted_duration[0]\n",
    "    else:\n",
    "        # 값이 있는 경우 그대로 반환\n",
    "        return row['통증기간(월)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "0f38b322-2c74-4546-8ed5-48e4f1cd2bb4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\02-19\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\02-19\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\02-19\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\02-19\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "df_ROS_concat['통증기간(월)'] = df_ROS_extract.apply(predict_missing_duration, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "cde9ce42-6794-4583-8d47-5cd19d332576",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_concat['통증기간(월)'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "914cc3e2-078f-46e1-9661-01d363e6fa90",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "환자통증정도              0\n",
       "입원기간                0\n",
       "통증기간(월)             0\n",
       "연령                  0\n",
       "체중                  0\n",
       "Large Lymphocyte    0\n",
       "수술기법                0\n",
       "수술기법_IELD           0\n",
       "수술기법_TELD           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_concat.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59767d99-f25c-44e7-ba7d-181a10cd6347",
   "metadata": {},
   "source": [
    "#### Scaling - MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "dbc8e5cd-6176-47a2-b4ac-9257a1aeacba",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['환자통증정도', '입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법',\n",
       "       '수술기법_IELD', '수술기법_TELD'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ROS_concat.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2365f56c-07ba-4801-8641-9cba0b4a1ea9",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "target = df_ROS_concat['환자통증정도']\n",
    "features = df_ROS_concat.drop(columns=['환자통증정도', '수술기법'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5be87e6b-b3a1-4e74-b309-ffcd47a8a0f5",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['입원기간', '통증기간(월)', '연령', '체중', 'Large Lymphocyte', '수술기법_IELD',\n",
       "       '수술기법_TELD'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "ea1bd2bf-2b1e-4324-8d87-ff2e83e78df7",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "minMaxScaler= MinMaxScaler()\n",
    "features = minMaxScaler.fit_transform(features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e99cd166-5f60-4752-aeac-69ad8fe3f17d",
   "metadata": {
    "tags": []
   },
   "source": [
    "#### Imbalanced Data Sampling\n",
    "-  Under Sampling : Tomek's Link "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e9e2befd-7e79-4f17-821f-3a417e29ff68",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from imblearn.under_sampling import TomekLinks\n",
    "from sklearn.datasets import make_classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "4325f28b-e8a5-4c32-90d6-b09aecfc7eaa",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "features, target = make_classification(n_classes=2, class_sep=2,\n",
    "                    weights=[0.3, 0.7], n_informative=3, n_redundant=1, flip_y=0,\n",
    "                    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "6930d0e1-745b-43cb-859a-bb382f9276d6",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1000, 20), (1000,))"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features.shape, target.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "a6b12b21-f559-4da4-be1b-ec043d7308a2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({0: 300, 1: 700})"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from collections import Counter\n",
    "Counter(target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "5eb349c3-ba99-4381-81ef-a64f294d337c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "tomekLinks = TomekLinks() #인스턴스화\n",
    "features_resaple, target_resample = tomekLinks.fit_resample(features, target) #학습"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b16992d3-749c-42d9-aae9-f4eb3c00ae6a",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((996, 20), (996,))"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features_resaple.shape, target_resample.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "4f2323d5-3124-454e-8bff-9d71c1526d5b",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({0: 300, 1: 696})"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Counter(target_resample)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "314d842f-a2e3-4f1c-8e79-5b112c18ae09",
   "metadata": {},
   "source": [
    "### 정형화"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96ae94fd-c7f5-4af5-8d34-c4fc464a000e",
   "metadata": {},
   "source": [
    "#### split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "713706c9-345c-42f2-8ab2-8611216a68b3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((750, 20), (750,), (250, 20), (250,))"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=111)\n",
    "features_train.shape, target_train.shape, features_test.shape, target_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8af3b99c-fa4e-42f7-a6b7-b4274068c886",
   "metadata": {},
   "source": [
    "### 모델 학습\n",
    "- 목표변수가 범주형일 경우 : DecisionTreeClassifier\n",
    "- 목표변수가 연속형일 경우 : DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "a8d8b3b3-51ed-4a54-b724-151a085f0775",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor\n",
    "model = DecisionTreeClassifier() \n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "d17a49ed-598a-4a6d-8b7e-69777ef846eb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "hyper_params = {'min_samples_leaf' : range(2,6) #[2,3,4,5,6]\n",
    "               ,'max_depth' :  range(2,6)\n",
    "               ,'min_samples_split' :  range(2,6)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "26d53922-98a0-4458-828f-431924884809",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#from sklearn.metrics import f1_score, make_scorer\n",
    "grid_search = GridSearchCV(model, param_grid=hyper_params, cv=3, verbose=1) # scoring=scoring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "50b462f1-c52e-4c48-ae25-b4ed55fd5ac6",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 64 candidates, totalling 192 fits\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=3, estimator=DecisionTreeClassifier(),\n",
       "             param_grid={&#x27;max_depth&#x27;: range(2, 6),\n",
       "                         &#x27;min_samples_leaf&#x27;: range(2, 6),\n",
       "                         &#x27;min_samples_split&#x27;: range(2, 6)},\n",
       "             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=3, estimator=DecisionTreeClassifier(),\n",
       "             param_grid={&#x27;max_depth&#x27;: range(2, 6),\n",
       "                         &#x27;min_samples_leaf&#x27;: range(2, 6),\n",
       "                         &#x27;min_samples_split&#x27;: range(2, 6)},\n",
       "             verbose=1)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=3, estimator=DecisionTreeClassifier(),\n",
       "             param_grid={'max_depth': range(2, 6),\n",
       "                         'min_samples_leaf': range(2, 6),\n",
       "                         'min_samples_split': range(2, 6)},\n",
       "             verbose=1)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search.fit(features_train, target_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "74ddfec0-2635-4c5d-b1e6-08f33c45c972",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" checked><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "1bad5495-574d-4016-8fe1-b0644ae00191",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.988, {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 3})"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search.best_score_, grid_search.best_params_ \n",
    "\n",
    "## best_score : 0.988 /정확도(accuracy) --> 높은 정확도로, 모델이 예측을 매우 잘 수행함"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "9479c841-3712-454a-b222-d6cb62bcd95c",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-7\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" checked><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=3)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_model = grid_search.best_estimator_ # 하나의 모델 --> 그 중에서 최고의 모델\n",
    "best_model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "a3f39adc-c21f-4c0f-a9b6-cc6555259e62",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,\n",
       "       1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0,\n",
       "       0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,\n",
       "       0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1,\n",
       "       1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,\n",
       "       0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,\n",
       "       0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
       "       1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,\n",
       "       0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,\n",
       "       1, 1, 1, 1, 1, 0, 1, 1])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target_test_predict = best_model.predict(features_test)\n",
    "target_test_predict"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7aa2958b-22b7-48ca-8292-03d50cdb5bf8",
   "metadata": {},
   "source": [
    "### 평가"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ecfbb531-52a1-4acb-b325-d43a51791557",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.99      0.99        76\n",
      "           1       0.99      0.99      0.99       174\n",
      "\n",
      "    accuracy                           0.99       250\n",
      "   macro avg       0.99      0.99      0.99       250\n",
      "weighted avg       0.99      0.99      0.99       250\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(target_test, target_test_predict))\n",
    "\n",
    "# support는 각 클래스에 속하는 샘플의 개수 -->  클래스 0은 76개의 샘플이 있고, 클래스 1은 174개의 샘플이 있음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d76266c-4457-4ac0-99d6-dde85372a7fb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
