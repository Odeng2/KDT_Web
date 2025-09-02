# # 예제 : 상속
# import datetime

# class BankAccount:
    
#     def __init__(self, ouner, balance):
#         self.owner = owner
#         self.balance = balance
#         self.transaction_history = []

#     def deposit(self, amount):
#         if amount <= 0:
#             return False
#         self.balance += amount
#         print(f"{amount}가 입금되었습니다.")
#         self._log_transaction("입금", amount)
#         return True
    
#     def withdraw(self, amount):
#         if amount > self.balance:
#             print("잔액이 부족합니다.")
#             return False
        
#         self.balance = 
#         print(f"{amount}가 출금되었습니다.")
#         self._log_transaction("입금", amount)
#         return True
    
#     def _log_transaction(self, transaction_type, amount):
#         timestamp = datetime.datetime.now().strftime('')
#         self.transaction_history.append({
#             "type": transaction_type,
#             "amount": amount,
#             "timestamp": timestamp,
#             "balance": self.balance
#         })

#     def print_transaction_history(elf):
#         for transaction in self.transaction_history:
#             print

# my_account = BankAccount("홍길동", 1000000)
# my_account.deposit(3000)
# my_account.withdraw(50000)
# my_account.print_transaction_history()



# # 예제 : 
# file = open('example.txt', 'r')
# contents = file.read()
# print(contents)


# # 8월 26일 예제 : 바이너리 파일 처리
# def copy_image(source_file, destination_file):
#     try:
#         with open(source_file, 'rb') as source:
#             image_data = source.read()

#         with open(destination_file, 'wb') as destination:
#             destination.write(image_data)
#         return True
    
#     except FileNotFoundError:
#         print(f"{source_file} 파일을 찾을 수 없음")
#         return False
#     except Exception as e:
#         print(f"에러 : {e}")
#         return False

# copy_image()


# # 예제 : 파일 읽어서 -> 암호화 -> 복호화
# def xor_encrypt_decrypt(input_file, output_file, key):
#     try: 
#         with open(input_file, 'rb') as infile:
#             data = infile.read
#         key_bytes = key.encode() if isinstance(key, str) else bytes([key])
#         key_len = len(key_bytes)

#         encrypted_data = bytearray(len(data))
#         for i in range(len(data)):
#             encrypted_data[i] = data[i] ^ key_bytes[i % key_len]

#         # data의 길이가 100, index의 범위는 0~99
#         # => encrypted_data의 기길이가 100
#         # mykey123의 길이 8, index의 길이는 0~7
#         # 주어진 index가 90이라면? 90 % 8 = 2. 91 % 8 = 3

#         with open(output_file, 'wb') as outfile:
#             outfile.write(encrypted_data)

#     except Exception as e:
#         print(f'오류 {e}')

# # 암호화
# xor_encrypt_decrypt('example.txt', 'secret.enc', 'mykey123')

# # 복호화
# xor_encrypt_decrypt('secret.enc', 'decrypted.txt', 'mykey123')



# # 예제 : 파일 인코딩
# import csv

# with open('data.csv', 'r', encoding='utf-8') as file:
#     header = file.readline().strip().split(',')
#     # strip(): 문자열 양쪽의 공백과 개행 문자를 제거함
#     print(header)

#     for line in file:
#         values = line.strip().split(',')
#         print(values)


# # 예제: 백업 스크립트 작성
# import os
# from pathlib import Path
# import datetime
# import zipfile

# def backup_directory(source_dir, backup_directory=None, backup_name=None):
#     source_path = Path(source_dir)
#     if not source_path.exists() or not source_path.is_dir():
#         print(f"{source_path} 디렉토리가 존재하지 않음" )
#         return False
    
#     if backup_directory is None:
#         backup_directory = Path.cwd()

#     else:
#         backup_directory = Path(backup_directory)
#         backup_directory.mkdir(parents=True, exist_ok=True)

#     if backup_name is None:
#         timestamp = datetime.datetime.now().strftime("%Y%m%d")
#         backup_name = f"{source_path.name}_backup_{timestamp}.zip"

#     backup_path = backup_directory / backup_name
#     with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(source_dir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 arc_name = os.path.relpathfile_path, os.path.dirname(source_dir)
#                 zipf.write(file_path, arc_name)


 
# 8월 27일 =======================================================

# # 예제 : 제너레이터
# def count_up_to(max):
#     count = 1
#     while count <= max:
#         yield count
#         count += 1

# counter = count_up_to(5)
# print(next(counter))
# print(next(counter))
# print(next(counter))


# # 이터레이터 + 제너레이터
# import time
# import random

# def sensor_data_stream():
#     while True:
#         temperature = 20 + random.uniform(-5, 5)
#         yield f"온도 : {temperature: .2f}, 시간 : {time.strftime('%H:%M:%S')}"
#         time.sleep(1)

# stream = sensor_data_stream()
# for _ in range(5):
#     # 함수 진행
#     print(next(stream))
#     # 함수 진행


# # Thread
# import threading
# import time

# def background_task():
#     while True:
#         print("백그라운드 작업 실행 중")
#         time.sleep(1)

# daemon_thread = threading.Thread(target=background_task, daemon=True)
# daemon_thread.start()

# print("메인 쓰레드 시작")
# time.sleep(3)
# print("메인 쓰레드 종료")


# # 스레드 동기화
# import threading
# import time

# event = threading.Event()

# def waiter():
#     print("대기자 : 이벤트를 기다리는 중")
#     event.wait()
#     print("대기자 : 이벤트 수신 및 작업 진행")

# def setter():
#     print("설정자 : 작업 중")
#     time.sleep(3)
#     print("설정자 : 설정 완료")
#     event.set()

# t1 = threading.Thread(target=waiter)
# t2 = threading.Thread(target=setter)

# t1.start()
# t2.start()


# # 스레드 - condition
# import threading
# import time

# data = None
# condition = threading.Condition()

# def wait_for_data():
#     print("대기 : 데이터 대기중")
#     with condition:
#         condition.wait()
#         print(f"대기 : 데이터 {data} 수신 완료")

# def prepare_data():
#     global data
#     print("준비 : 데이터 준비중")
#     time.sleep(2)
#     with condition : 
#         data = "준비된 데이터"
#         print("준비 : 데이터 준비 완료")
#         condition.notify()

# t1 = threading.Thread(target=wait_for_data)
# t2 = threading.Thread(target=prepare_data)

# t1.start()
# t2.start()

# t1.join()
# t2.join()


# # 작업 분베
# import threading
# import time
# import queue
# import random

# task_queue = queue.Queue()
# result_queue = queue.Queue()

# def create_tasks():
#     print("작업 생성 시작")
#     for i in range(10):
#         task = f"작업-{i}"
#         task_queue.put(task)
#         print(f"작업 '{task}' 추가 됨")
#         time.sleep(random.uniform(0.1, 0.3))
#     for _ in range(3):
#         task_queue.put(None)
#     print('모든 작업 생성 완료')

# def worker(worker_id):
#     print(f"워커 : {worker_id} 시작")
#     while True:
#         task = task_queue.get()
#         if task is None:
#             print(f"워커 : {worker_id} 작업 종료")
#             break

#         # 작업 코드
#         print(f"워커 {worker_id}가 {task} 처리 중...")
#         processing_time = random.uniform(0.5, 1.5)
#         time.sleep(processing_time)

#         result = f"{task} 완료 (소요시간: {processing_time:.2f}초)"
#         # 작업 완료
#         result_queue.put(worker_id, result)
#         task_queue.task_done()
#         print(f"남은 작업 수 : {task_queue.qsize()}")

# def result_collector():
#     print("결과 수집기 시작")
#     results = []

#     for _ in range(10):
#         worker_id, result = result_queue.get()
#         print(f"결과 수집 워커 {worker_id} -> {result}")
#         results.append(result)
#         result_queue.task_done()

#     print(f"결과 {len(results)} 처리 완료")

# creator = threading.Thread(target=create_tasks)
# workers = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
# collector = threading.Thread(target=result_collector)

# creator.start()
# for w in workers:
#     w.start()
# collector.start()

# creator.join()
# for w in workers:
#     w.join()
# collector.join()

# print("모든 작업 완료!")



# # ===========================
# import threading
# import time
# import queue
# import random
# import concurrent.futures

# def task(params):
#     name, duration = params
#     print(f"작업 {name} 시작")
#     time.sleep(duration)
#     return f"{name} 완료, 소요시간 : {duration}초"

# params = [
#     ("A", 2),
#     ("B", 1),
#     ("C", 3),
#     ("D", 1.5)
# ]

# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     results = list(executor.map(task, params))
#     for result in results:
#         print(result)



# # 프로세스 - lock 사용해서 동기화하기
# import multiprocessing
# import time

# def count_up(name, max_count):
#     for i in range(1, max_count + 1):
#         print(f"프로세스 {name}: 카운트 {i}")
#         time.sleep(0.5)

# def add_to_shared(shared_value, lock, increment):
#     for _ in range(5):
#         with lock:
#             shared_value.value += increment
#         time.sleep(0.1)
#     print(f"프로세스 {multiprocessing.current_process().name}: 작업 완료")

# if __name__ == "__main__":
#     lock = multiprocessing.Lock()
#     shared_number = multiprocessing.Value('i', 0)

#     p1 = multiprocessing.Process(target=add_to_shared, args = (shared_number, lock, 1))
#     p2 = multiprocessing.Process(target=add_to_shared, args = (shared_number, lock, 2))

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

#     print(f"shared num: {shared_number.value}")
#     print("모든 프로세스 종료!")



# # 프로세스 - Queue 활용하기
# import multiprocessing
# import time
# import random

# def producer_process(queue):
#     print(f"생산자 프로세스 시작 : {multiprocessing.current_process().name}")
#     for i in range(5):
#         item = f"데이터-{i}"
#         queue.put(item)
#         print(f"생산 : {item}")
#         time.sleep(random.uniform(0.1, 0.5))
#     queue.put(None)

# def consumer_process(queue):
#     print(f"소비자 프로세스 시작 {multiprocessing.current_process().name}")
#     while True:
#         item = queue.get()
#         if item is None:
#             break
#         print("소비 : {item}")
#         time.sleep(random.uniform(0.2, 0.7))
#     print("소비자 프로세스 종료")


# if __name__ == "__main__":
#     q = multiprocessing.Queue()

#     p1 = multiprocessing.Process(target=producer_process, args = (q, ))
#     p2 = multiprocessing.Process(target=consumer_process, args = (q, ))

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

#     print("모든 프로세스 종료!")



# # 프로세스 - pool 사용하기 (수정하기)==================================
# import multiprocessing
# import time
# import random

# def producer_process(queue):
#     print(f"생산자 프로세스 시작 : {multiprocessing.current_process().name}")
#     for i in range(5):
#         item = f"데이터-{i}"
#         queue.put(item)
#         print(f"생산 : {item}")
#         time.sleep(random.uniform(0.1, 0.5))
#     queue.put(None)

# def consumer_process(queue):
#     print(f"소비자 프로세스 시작 {multiprocessing.current_process().name}")
#     while True:
#         item = queue.get()
#         if item is None:
#             break
#         print("소비 : {item}")
#         time.sleep(random.uniform(0.2, 0.7))
#     print("소비자 프로세스 종료")


# if __name__ == "__main__":
#     num_cores = multiprocessing.cpu_count()
#     print(f"cores : {num_cores}")
#     tasks = range(10)

#     start_time = time.time()
#     sequential_results = [process_task(i) for i in range(tasks)]
#     end_time = time.time()
#     print(f"순차 처리 : {end_time - start_time:.2f}")

#     start_time = time.time()
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         parallel_results = pool.map(process_task, tasks)
#     end_time = time.time()
#     print(f"병렬 처리 : {end_time - start_time:.2f}")

#     p1 = multiprocessing.Process(target=producer_process, args = (q, ))
#     p2 = multiprocessing.Process(target=consumer_process, args = (q, ))

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

#     print("모든 프로세스 종료!")



# # 비동기 프로그래밍 (수정하기)
# import 




# # 웹 사이트 여러 개에서 동시에 정보 가져오기 (수정하기)
# import asyncio
# import aiohttp
# import time

# websites = [
#     "https://www.naver.com/"
# ]

# async def fetch(session, url):
#     print(f"{url} 요청 시작")
#     try: 
#         start_time = time.time()

#         async with session.get(url, timeout=10) as response:
#             content = await response.text()
#             elapsed = time.time() - start_time
#             print(f"{url} 응답 완료 : {len(content)} 바이트 (소요 시간: {elapsed})")
#             return url, len(content), elapsed
#     except Exception as e:
#         print(f"{url} 오류 발생 : {e}")
#         return url

# async def fetch_all_sequential(urls):
#     start_time = time.time()
#     results = []

#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch(session, url) for url in websites]
#         await asyncio.gather(*tasks)

#         for url in urls:
#             result = await fetch(session, url)
#             results.append(result)
    
#     end_time = time.time()
#     print(f"순차 처리 완료: {end_time - start_time:.2f}초 소요")
#     return results

   



# async def main():
#     await asyncio.sleep(1)
#     sequential_results = await fetch_all_sequential(websites)

#     if __name__ == "__main__":
#         main()


# # ========================================================


