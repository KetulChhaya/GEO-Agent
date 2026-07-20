from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="CiteSight")
app.include_router(router)
