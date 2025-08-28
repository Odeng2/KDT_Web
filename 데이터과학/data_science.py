# # 8월 27일 (수) ==============================================================
# import numpy as np

# print(np.__version__)

# arr = np.array([1, 2, 3, 4])
# print(arr)

# zeros = np.zeros((3, 4))
# print(zeros)

# ones = np.ones((2, 3))
# print(ones)

# range_arr = np.arange(0, 10, 2)
# print(range_arr)

# linear_space = np.linspace(0, 1, 5)
# print(linear_space)

# random_arr = np.random.random((2, 2))
# print(random_arr)



# 8월 28일 (목) ========================================
import numpy as np

# arr = np.array([[1, 2, 3], [4, 5, 6]])

# # 주요 속성
# print(f"배열 차원: {arr.ndim}")
# print(f"배열 형태: {arr.shape}")
# print(f"배열 크기: {arr.size}")
# print(f"요소 데이터 타입: {arr.dtype}")
# print(f"각 요소 바이트 크기: {arr.itemsize}")
# print(f"전체 배열 바이트 크기: {arr.nbytes}")

# # 행과 열의 차원 바꾸기
# print(arr.T)

# # 1차원 배열을 2차원 배열로 바꾸기
# arr1d = np.arange(12)
# print(arr1d)
# arr2d = arr1d.reshape(3, 4)
# print(arr2d)

# # 다시 평탄화됨
# print(arr2d.flatten())

# # 데이터 타입 변경하기
# arr_float = arr.astype(np.float64)
# print(arr_float.dtype)

# # 값 계산하기
# data = np.array([1, 2, 3, 4, 5])
# print(f"합계: {data.sum()}")
# print(f"평균: {data.mean()}")
# print(f"최소값: {data.min()}")
# print(f"최대값: {data.max()}")
# print(f"표준편차: {data.std()}")
# print(f"분산: {data.var()}")
# print(f"누적합: {data.cumsum()}")

# # 원하는 값 가져오기
# print(data[data%2==0])

# # 행렬곱
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])

# print(a.dot(b))
# print(a*b)    # 행렬곱과 요소별 곱셈은 다르다!

# # 인덱싱
# arr_ex1 = np.array([1, 2, 3, 4, 5])
# print(arr_ex1[0])
# print(arr_ex1[-1])
# print(arr_ex1[:3])
# print(arr_ex1[::2])

# arr_ex2 = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# print(arr_ex2[0, 1])
# print(arr_ex2[:2])
# print(arr_ex2[:2, 1:3])

# # 배열끼리의 연산
# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array([5, 6, 7, 8])
# print(b - a)

# # 스칼라 연산
# print(arr1 + 10)
# print(arr1 * 10)
# print(arr1**2)

# # reshape 메소드
# arr3 = np.arange(12)
# reshaped1 = arr3.reshape(2, 6)
# reshaped2 = arr3.reshape(3, -1)    # 행이 3개가 되고, 나머지는 알아서 계산됨
# reshaped1 = arr3.reshape(-1, 2)
# print(reshaped1)

# # 요소 합치기
# arr11 = np.array([[1, 2], [3, 4]])
# arr22 = np.array([[5, 6], [7, 8]])

# vertical = np.concatenate([arr11, arr22], axis=0)
# print(vertical)
# v_stack = np.vstack([arr11, arr22])
# print(v_stack)

# horizontal = np.concatenate([arr11, arr22], axis=1)
# print(horizontal)
# h_stack = np.hstack([arr11, arr22])
# print(h_stack)

# #
# arr33 = np.arange(12).reshape(3, 4)
# print(arr33)
# v_split = np.split(arr33, 3, axis=0)
# print(v_split)
# for i, split_arr in enumerate(v_split):
#     print(i)
#     print(split_arr)

# # 필터 : 원하는 데이터 추출하기
# ages = np.array([23, 18, 45, 61, 17, 34, 57, 28, 15])

# adult_filter = ages >= 18
# adults = ages[adult_filter]
# print(adults)

# young_filter = (ages >=18) & (ages < 30)
# young_people = ages[young_filter]
# print(young_people)

# ticket_prices = np.zeros_like(ages)    # ages와 똑같은 크기의 배열이 생성됨.
# print(ticket_prices)
# ticket_prices[ages<18] = 5
# ticket_prices[(ages>=18) & (ages<60)] = 10
# ticket_prices[ages >= 60] = 8
# print(ticket_prices)

# #
# import matplotlib
# import matplotlib.pyplot as plt

# image = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 0, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0]
# ])
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title('original')

# # 이미지를 밝게 조정하기
# brightened = image + 0.5
# print(brightened)
# plt.subplot(1, 3, 2)
# plt.imshow(brightened, cmap='gray', vmin=0, vmax=1)
# plt.title('brightened')

# # 이미지 반전
# inverted = 1 - image
# plt.subplot(1, 3, 3)
# plt.imshow(inverted, cmap='gray', vmin=0, vmax=1)
# plt.title('inverted')

# plt.tight_layout()
# plt.show()


# pandas ===============================================================

import pandas as pd
# print(pd.__version__)

# # Series: 인덱스 지정 가능 -> 딕셔너리 파싱에 유용함
# s = pd.Series([1, 3, 4, 5], index=['a', 'b', 'c', 'd'])
# print(s)                                                                                                                                                                                                                                                                                                                     

# population = {
#     'Seoul': 9776,
#     'Busan': 3429,
#     'Incheon': 2947,
#     'Daegu:': 6016
# }
# pop_series = pd.Series(population)
# print(pop_series)

# s1 = pd.Series([10, 20, 30, 40, 50], index=['a','b','c','d','e'])
# print(s1.values)
# print(s1.index)
# print(s1.mean())
# print(s1.sum())
# print(s1.min())
# print(s1.max())
# # 데이터 접근 방법
# print(s1['c'])
# print(s1[['a', 'b', 'c']])
# print(s1[s1>30])
# print(s1.apply(np.sqrt))

# # T/F값 받기
# s2 = pd.Series([10, np.nan, 30, np.nan, 50], index = ['a','b','c','d','e'])
# print(s2.isna())
# # 숫자 아닌 값 날리기
# print(s2.dropna())
# # 숫자 아닌 값을 다른 숫자로 채우기
# print(s2.fillna(0))

# DataFrame
data = {
    'name': ['john', 'anna', 'peter', 'bob'],
    'age': [28, 24, 35, 32],
    'city': ['NewYork', 'Paris', 'Beriln', 'London'],
    'salary': [5000, 6400, 7500, 8000]
}
df = pd.DataFrame(data)
print(df)

# DataFrame 속성
print("크기(행, 열):" , df.shape)
print("열 이름:" , df.columns)
print("행 인덱스:" , df.index)
print("데이터 타입:" , df.dtypes)

# DataFrame 메소드
print("처음 2행\n:", df.head(2))
print("마지막 2행\n:", df.tail(2))
print("기본 통계량\n:", df.describe())

# 데이터 접근하기
print("'Age' 열: \n", df['age'])
print("여러 열 선택: \n", df[['name', 'salary']])
print("첫 3행: \n", df.iloc[0:3])
print("조건부 선택: \n", df[df['age'] > 30])

# 값 수정하기
df['age'] = df['age'] + 1
print(df)

# 열 추가하기 (df에서 열은 정의되어 있는 기준으로 세로값)
df['country'] = ['USA', 'France', 'Germany', 'UK']
# 행 추가하기
df.loc[4] = ['Charlie', 29, 'Sydney', 70000, 'Austrailia']
# 열 삭제하기
df.drop('country', axis=1, inplace=True)
# 행 삭제하기
df.drop(4, axis=0, inplace=True)      
print(df)

print(df.loc[0:1])
print(df.loc[0:1, ['name', 'age']])

# DF - GroupBy
