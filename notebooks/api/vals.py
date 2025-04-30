# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class Movies(BaseModel):
    summary:str
