# 1. Library imports
import uvicorn
from fastapi import FastAPI
from vals import Movies
from my_functions import search_movies

# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.0:8000
@app.get('/')
def index():
    return {'message': 'Welcome, you can predict movie gender'}

@app.post('/predict')
def predict(data:Movies):
    data = data.dict()
    texto = data['summary']
    out_model = search_movies(query=texto)
    return {out_model}

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8989)
