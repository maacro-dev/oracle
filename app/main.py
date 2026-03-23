from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionRequest, PredictionResponse
from .predictor import run_predictions
from .config import settings

app = FastAPI(title="Yield Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest = PredictionRequest()):
    """
    Run yield predictions for a given season (or the latest season).
    Optionally limit to specific fields by providing a list of mfid_ids.
    """
    try:
        result = await run_predictions(
            season_id=request.season_id,
            mfid_ids=request.mfid_ids
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

