import 
from typing import 
from pydantic import BaseModel

router = APIRouter()

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True

class UserCreate(BaseModel):
    name: str
    email: str

fake_users_db = [
    User(id=1, name="Alice", email="alice@example.com")
    User(id=2, name="Bob", email="bob@example.com")
]


# CRUD
@router.get('/', response_model = List[User])
def read_users(skip:int=0, limit:int=100):
    return fake_users_db[skip:skip+limit]

