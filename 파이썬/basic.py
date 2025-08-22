## 예제 #1:
# def greet (name, time_of_day='아침'):    #time_of_day의 기본값: 아침
#     greetings={
#         '아침': '좋은 아침입니다',
#         '점심': '맛있는 점심 드세요.'
#     }
#     return f"{greetings[time_of_day]}, {name}님!"

# print(greet('dahan'))


# # 예제 #2:
# text = "Python"
# first_char = text[0]   #'P'
# lasg_char = text[-1]    #'n' 음수 인덱스는 문자열 끝에서부터 시작
# #문자열 슬라이싱
# substring = text[1:4]
# print(substring)


# # 8/21 예제 #1:
# print(bool(0))
# print(bool(-1))
# print(bool(""))
# print(bool([]))


# # 예제 #2:
# x = True
# y = False
# print(x and y)
# print(x or y)
# print(not x)


# # 예제 #3:
# print(False and print("확인"))    # False and까지 보고 뒤의 구문이 무시됨.
# print(True or print("확인"))    # True or 이므로 뒤에 적은 구문이 무시됨.


# # 예제 #4:
# print(0 and 5)
# print(2 and 5)
# print(0 or 5)
# print(2 or 5)


# # 예제 #5: 비교 연산자
# print(1 == 2)
# print(1 != 2)
# print(5 > 2)
# print(5 < 2)


# # 예제 #6:
# list1 = [1, 2, 3]
# list2 = [1, 2, 3]
# lit3 = list1
# print(list1 == list2)
# print(list1 is list2)   # 객체비교: 같은 값이지만 객체는 다름.
# print(list1 is list3)   # 객체비교 : 같은 주소 참조.


# # 예제 #7: 배열
# numbers = [x**2 for x in range(1, 6)]
# print(numbers)


# # 예제 #8: todo 리스트
# tasks = []


# def add_task(task): 
#     tasks.append(task)
#     print(f"{task}가 추가 되었습니다.")

# def complete_task(task_index):
#     completed = tasks.pop(task_index)
#     print(f"'{completed}'를 완료함")


# def view_tasks():    
#     for i, task in enumerate(tasks):
#         print(f"{i} : {task}")

# add_task('파이썬 마스터')
# add_task('운동하기')
# view_tasks()
# complete_task(1)
# view_tasks()


# # 예제 #9: 중첩 리스트
# nested = [[1, 2, 3], [4, 5, 6]]

# # 반복문
# rows, cols = 3, 4
# matrix = []
# for i in range(rows):
#     row = []
#     for j in range(cols):
#         row.append(i*cols+j)
#     matrix.append(row)

# print(matrix)

# # 중첩 리스트 한 줄로 만들기
# matrix2 = [[i*cols+j for j in range(cols)] for i in range(rows)]
# print(matrix2)


# # 예제 10: 결과 예측해보기 Quiz
# def modify_list(lst):
#     lst.append(4)
#     lst = [7, 8, 9]
#     return lst

# original = [1, 2, 3]
# result = modify_list(original)
# print(original)
# print(result)


# # 예제 11:
# coordinates = (10, 20)

# numbers = tuple([1, 2, 3, 4, 5])

# # 중요: 이렇게 표현하면 튜플로 저장된다!!!
# colors = "red", "green", "blue"

# single_item = (42, )    # 튜플
# not_tuple = (42)    # 튜플 아님

# # 빈 튜플 만들기
# empty_tuple = ()
# empty_tuple2 = tuple()

# mixed_tuple = (1, 'dsdd', True)


# # 예제 12: 언패킹
# rgb = (255, 100, 50)
# red, green, blue = rgb
# print(red)
# print(green)
# print(blue)

# numbers = (1, 2, 3, 4, 5)
# first, *middle, last = numbers
# print(first)
# print(middle)
# print(last)

# def get_user_info():
#     return '홍길동', 30, '서울'    # 리턴값이 콤마(,)로 이루어져 있으면 튜플로 반환됨.
# print(get_user_info())
# name, age, city = get_user_info()
# print(name)


# # 예제 13:
# import time

# data_size = 1000000
# search_key = f"key_{data_size-1}"

# list_data = [(f"key_{i}", i) for i in range(data_size)]
# dict_data = {"key_{i}" : i for i in range(data_size)}

# start_time = time.time()
# result_list = None
# for k, v in list_data:
#     if k == search_key:
#         result_list = v
#         break
# list_time = time.time() - start_time


# # 예제 14:
# student = {"name":"홍길동", "age":20}
# student2 = dict(name="김철수", age=22)
# items = [("name", "이영희"), ("age", 20)]
# student3 = dict(items)

# empty_dict = {}
# empty_dict2 = dict()

# keys = ["name", "age"]
# student4 = dict.fromkeys(keys, "미정")    #해당 키의 값들은 모두 미정임.


# # 예제 15:
# student = {
#     "name": "홍길동",
#     "age": 20,
#     "course": ["수학", "영어", "과학"]
# }

# new_info = {"grade":"A", "age":21}
# student.update(new_info)
# print(student)

# student_copy = student.copy()

# student.clear()


# # 예제 16:
# users = {
#     "user1": {
#         "name": "홍길동",
#         "age": 30,
#         "course": ["수학", "영어", "과학"]
#     },
#     "user2": {
#         "name": "김철수",
#         "age": 25,
#         "course": ["사회", "역사"]
#     }
# }

# print(users["user1"]["name"])
# for user_id, user_info in users.items():
#     print(f"user_id : {user_id}")
#     for key, value in user_info.items():
#         print(f"{key}:{value}")


# # 예제 17: 
# squares = {x:x**2 for x in range(1, 6)}
# print(squares)

# fruits = ['apple', 'banana', 'cherry']
# fruitsDict = {fruit:len(fruit) for fruit in fruits}
# print(fruitsDict)


# # 예제 18:
# import requests

# url = 'https://jsonplaceholder.typicode.com/posts/1'
# response = requests.get(url)
# # print(response)
# if response.status_code == 200:
#     data = response.json()
#     # print(data)
    
#     print(f"게시물 ID: {data['id']}")
#     print(f"제목 : {data['title']}")
#     print(f"내용: {data['body']}")
#     print(f"작성자 ID: {data['userId']}")
    

# # 8월 22일 예제 #1: 집합
# fruits = {"사과", "바나나", "체리"}
# numbers = set ([1, 2, 3, 2, 1])
# print(numbers)
# chars = set("hello")
# print(chars)
# empty_set = set()
# not_empty_set = {}   # 빈 디겨너리. 집합이 아님.

# squares = {x**2 for x in range(1, )}
# print(squares)



# # 예제 2:
# fruits = {"사과", "바나나", "체리"}
# fruits.add('딸기')
# fruits.update(['망고', '블루베리'])
# print(fruits)
# fruits.remove('바나나')
# fruits.remove('바나나2')
# fruits.discard('딸기2')
# print(fruits)
# popped = fruits.pop()
# print(f'popped : {popped}')
# fruits.clear()
# print(fruits)


# # 예제 3: 
# numbers = {5, 4, 3, 2, 1}
# for num in numbers:
#     print(num)

# for num in sorted(numbers):
#     print(num)


# # 예제 4: 합집합, 교집합, 차집합
# class_a_hobbies = {"축구", "농구", "독서", "게임", "요리"}
# class_b_hobbies = {"야구", "농구", "독서", "그림", "요리"}

# common_hobbies = class_a_hobbies & class_b_hobbies
# print(common_hobbies)

# only_a_hobbies = class_a_hobbies - class_b_hobbies
# print(only_a_hobbies)

# all_hobbies = class_a_hobbies | class_b_hobbies
# print(all_hobbies)


# # 예제 5:
# numbers = [1, 2, 3, 1, 2, 123, 1, 2, 1, 3]
# unique_numbers = list(set(numbers))
# print(unique_numbers)

# result = []
# unique_set = set()
# for num in numbers:
#     if num not in unique_set:
#         unique_set.add(num)
#         result.append(num)


# # 예제 6:
# a = True
# b = Falsec = True

# print(a and b or c)
# print(a and (b or c))


# # 예제 7:
# result = 2**3*4+5
# print(result)


# # 예제 8:
# result = True or False and not True
# print(result)


# # 예제 9:
# rain_forecast = True
# umbrella_at_home = False

# if rain_forecast:
#     print("비가 올 예정입니다.")
#     if umbrella_at_home:
#         print("우산을 가지고 나갑니다.")
#     else:
#         print("우산을 구매해야 합니다.")
    

# # 예제 10: 삼항 연산자
# x = -10

# abs_value = x if x>=0 else -x
# test_result = '합격' if score>=70 else '불합격'

# def abs_fuc(x):
#     if x>=0:
#         return x
#     else:
#         return -x
    

# # 예제 11:
# temperature = 25

# message = '더운 날씨입니다' if temperature>30 else "적당한 날씨입니다" if temperature>20 else '추운 날씨입니다'
# print(message)


# 예제 12: lambda
numbers = [1, 2, 3, 4, 5]
squars = list(map(lambda x:x**2, numbers))
print(squars)