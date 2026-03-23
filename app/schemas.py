from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    season_id: Optional[int] = None      # If None, use latest season
    mfid_ids: Optional[List[int]] = None # If provided, predict only these fields
    force: bool = False                  # Force recompute

class PredictionResponse(BaseModel):
    season_id: int
    total_fields: int
    predicted_count: int
    message: str
