#
from fastapi import FastAPI, Query
import uvicorn
from typing import Optional, List



app = FastAPI()


# # 기본 예제 =================================================
# @app.get('/')
# def read_root():
#     return {'message': 'FastAPI 호출 완료'}

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=8000)


# # 선언적 라우팅 (데코레이터로 함수 동작 선언) ==========================
# @app.get('/menu')
# def get_menu():
#     return {"메뉴": ["커피"]}

# @app.post('/orders')
# def create_order():
#     return {'message': 'order created'}


# # fastapi http메소드 활용 =====================================
# @app.get('/menu')
# def get_menu():
#     return {"메뉴": ["아메리카노", "라떼", "카푸치노"]}


# @app.post("/orders")
# def create_order(order_data: dict):
#     return {"메시지": "주문 완료", "주문번호": 123}


# @app.put("/orders/123")
# def update_order(order_data: dict):
#     return {"메시지": "주문이 수정되었습니다."}


# @app.patch("/orders/123")
# def patch_order(partial_data: dict):
#     return {"메시지": "주문 옵션이 변경되었습니다."}


# @app.delete("/orders/123")
# def cancel_order():
#     return {"메시지": "주문이 취소되었습니다."}


# # 경로 매개변수 ==============================================
# @app.get('/users/{user_id}')
# def get_user(user_id: int):
#     return {'user id': user_id}


# @app.get('/cafe/{menu_type}/{item_name}')
# def get_menu_item(menu_type: str, item_name: str):
#     return {
#         'category': menu_type,
#         'item': item_name,
#         'message': f"{menu_type} 카테고리의 {item_name}을 선택하셨습니다."
#     }



# # 쿼리스트링 ================================================
# @app.get('/products')
# def get_products(
#     category: Optional[str] = None,
#     min_price: Optional[float] = None,
#     max_price: Optional[float] = None,
#     tags: List[str] = Query(default=[]),
#     in_stock: bool = True
# ):
#     filters = {
#         'category': category,
#         'price_range': f"{min_price} ~ {max_price}",
#         'tags': tags,
#         'in_stock': in_stock
#     }

#     return {"filters": filters, "message": "필터링된 상품 목록"}



# # 쿼리클래스 ===================================================
# @app.get('/restaurant/order')
# def order_food(
#     age: int = Query(..., age=18, le=65, description='주문자 나이'),
#     people: int = Query(2, gt=0, lt=11, description='식사 인원'),
#     budget: float = Query(None, ge=100000, description='예산 (원)')
# ):
#     return {
#         "주문정보": f"나이: {age}, 인원: {people}, 예산: {budget}"
#     }


# # 문자 제약 조건 걸기
# @app.get('/user/register')
# def register_user(
#     username: str = Query(..., min_length=3, max_length=20, description="사용자명")
# ):
#     return {"등록정보": {"username": username}}


# # 쿼리스트링으로 작성된 코드를 쿼리클래스를 활용해서 개선하기
# @app.get('/products-advanced')
# def get_products_advanced(
#     category: Optional[str] = Query(None, description="상품 카테고리"),
#     min_price: Optional[float] = Query(None, ge=0, description="최소 가격 (0 이상)"),
#     max_price: Optional[float] = Query(None, ge=0, le=100000, description="최대 가격"),
#     tags: List[str] = Query([], description="상품 태그 (복수 선택 가능)"),
#     in_stock: bool = Query(True, description="재고 여부")
# ):
#     return {
#         'category': category,
#         'min_price': min_price,
#         'max_price': max_price,
#         'tags': tags,
#         'in_stock': in_stock
#     }




# # pydantic 모델 활용 =====================================================
# from pydantic import BaseModel


# class Order(BaseModel):
#     menu_item: str
#     quantity: int
#     customer_name: str
#     special_request: str = '없음'


# @app.post('/orders')
# def create_order(order: Order):
#     return {
#         "message": "주문 접수"
#         "order_details": {
#             "메뉴": order.menu_item,
#             "수량": order.quantity,
#             "고객명": order.customer_name,
#             "특별요청": order.special_request
#         },
#         "order_id": 12345
#     }



# # pydantic - 응답 모델 선언하기 (반환값 검증 후 응답) ========================
# from pydantic import BaseModel
# from typing import List


# class Order(BaseModel):
#     order_id: int
#     customer_name: str
#     total_amount: float


# class OrderList(BaseModel):
#     orders: List[Order]
#     total_count: int


# @app.get("/orders", response_model=OrderList)
# def get_orders():
#     return {
#         "orders": [
#             {"order_id": 1, "customer_name": "김철수", "total_amount": 15000}
#         ],
#         "total_count": 1
#     }


# @app.post('/orders', response_model=Order, status_code=201)
# def create_order(order_data: dict):
#     return {"order_id": 123, "customer_name": "이영희", "total_amount": 25000}






