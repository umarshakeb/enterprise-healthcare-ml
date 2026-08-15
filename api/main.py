from fastapi import FastAPI
from api.routers import risk,claim,monitoring
from fastapi.routing import APIRoute

print("main.py loaded")
print("risk module: ", risk)
print("claim module: ", claim)

app = FastAPI(title="Healthcare ML API")

@app.get("/")
def root():
    return {"message": "Healthcare ML API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(risk.router, prefix="/predict", tags=['Risk Score Prediction'])
app.include_router(claim.router, prefix="/predict", tags=['Claim Status Prediction'])
app.include_router(monitoring.router, prefix="/monitor", tags=['Monitoring'])

print("Registered API routes:")
for route in app.router.routes:
    print(route)