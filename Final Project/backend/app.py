# Main FastAPI or Flask app for backend
from fastapi import FastAPI
from backend.api.routes import router

app = FastAPI(title="LLM Response Visualizer")

# Include routes from the API
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Welcome to the LLM Response Visualizer!"}
