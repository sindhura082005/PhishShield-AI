from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .inference import predict

app = FastAPI(title="Ref-TABMNet API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class URLPayload(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_url(payload: URLPayload):
    return predict(payload.url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
