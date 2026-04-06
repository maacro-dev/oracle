import pandas as pd
import numpy as np
from typing import List, Optional
from supabase import Client

from .config import settings
from .database import get_supabase_client
from .loader import load_model

NUMERIC_FEATURES = [
    'total_field_area_ha', 'seedling_age_at_transplanting',
    'distance_between_plant_row_1', 'distance_between_plant_row_2',
    'distance_between_plant_row_3', 'distance_within_plant_row_1',
    'distance_within_plant_row_2', 'distance_within_plant_row_3',
    'seeding_rate_kg_ha', 'num_plants_1', 'num_plants_2', 'num_plants_3',
    'average_number_of_plants', 'monitoring_field_area_sqm', 'applied_area_sqm',
    'nitrogen_content_pct_1', 'phosphorus_content_pct_1', 'potassium_content_pct_1',
    'amount_applied_1', 'amount_applied_2', 'amount_applied_3',
]

CATEGORICAL_FIELDS = [
    'barangay', 'municity', 'province', 'rice_variety', 'seed_class',
    'ecosystem', 'current_field_condition', 'crop_status', 'cause',
    'fertilizer_type_1', 'fertilizer_type_2', 'fertilizer_type_3',
    'crop_stage_on_application_1', 'crop_stage_on_application_2',
    'crop_stage_on_application_3', 'type_of_irrigation', 'source_of_irrigation',
    'crop_planted', 'soil_type', 'est_crop_establishment_method',
    'actual_crop_establishment_method', 'direct_seeding_method',
    'harvesting_method', 'irrigation_supply', 'gender'
]

def fetch_fields_for_prediction(
    supabase: Client,
    season_id: int,
    mfid_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Fetch field data for the given season from the flattened view.
    Only rows with harvest_date IS NULL are considered.
    """
    query = supabase.table("flattened_field_data")\
        .select("*")\
        .eq("season_id", season_id)

    if mfid_ids:
        query = query.in_("mfid_id", mfid_ids)

    response = query.execute()
    data = response.data
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

def preprocess_data(df_raw: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Transform raw field data into the feature matrix expected by the model.
    """
    X = pd.DataFrame(0, index=df_raw.index, columns=expected_features)

    # Fill numeric features
    for col in NUMERIC_FEATURES:
        if col in expected_features and col in df_raw.columns:
            X[col] = df_raw[col].fillna(0)

    # One-hot encode categorical fields
    for field in CATEGORICAL_FIELDS:
        mask = df_raw[field].notna()
        values = df_raw.loc[mask, field].astype(str)
        for idx, value in values.items():
            col_name = f"{field}_{value}"
            if col_name in expected_features:
                X.loc[idx, col_name] = 1

    # Boolean features
    if 'is_transplanted' in expected_features:
        X['is_transplanted'] = (df_raw['actual_crop_establishment_method']
                                 .str.contains('transplant', case=False, na=False)).astype(int)
    if 'has_damage' in expected_features:
        X['has_damage'] = (df_raw['cause'].notna()).astype(int)

    # Fill any remaining NaNs
    X = X.fillna(0)
    return X

def predict_and_store(
    supabase: Client,
    season_id: int,
    df_raw: pd.DataFrame,
    model,
    expected_features: List[str]
) -> int:
    """
    Run prediction and upsert results into the predicted_yields table.
    Returns the number of predictions stored.
    """
    if df_raw.empty:
        return 0

    X = preprocess_data(df_raw, expected_features)
    predictions = model.predict(X)

    records = []
    for i, row in df_raw.iterrows():
        records.append({
            "mfid_id": row["mfid_id"],
            "season_id": season_id,
            "predicted_yield_t_ha": float(predictions[i]) / 1000,
        })

    # Upsert in batches
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("predicted_yields").upsert(batch, on_conflict="mfid_id,season_id").execute()

    return len(records)

async def run_predictions(
    season_id: Optional[int] = None,
    mfid_ids: Optional[List[int]] = None
) -> dict:
    """
    Orchestrate the entire prediction pipeline.
    If season_id is None, automatically determine the latest season.
    """
    supabase = get_supabase_client()

    # Determine season_id
    if season_id is None:
        # Fetch the latest season
        resp = supabase.table("seasons").select("id").order("start_date", desc=True).limit(1).execute()
        if not resp.data:
            raise ValueError("No seasons found in the database.")
        season_id = resp.data[0]["id"]

    # Fetch data
    df_raw = fetch_fields_for_prediction(supabase, season_id, mfid_ids)
    if df_raw.empty:
        return {
            "season_id": season_id,
            "total_fields": 0,
            "predicted_count": 0,
            "message": "No eligible fields for prediction."
        }

    # Load model
    model, expected_features = load_model()

    # Predict and store
    predicted_count = predict_and_store(supabase, season_id, df_raw, model, expected_features)

    return {
        "season_id": season_id,
        "total_fields": len(df_raw),
        "predicted_count": predicted_count,
        "message": f"Predicted {predicted_count} fields for season {season_id}."
    }
