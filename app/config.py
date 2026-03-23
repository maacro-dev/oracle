import os
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

class Settings:
    @property
    def ALLOWED_ORIGINS(self):
        if not self.ALLOWED_ORIGINS_RAW:
            return ["http://localhost:3000"]
        return [o.strip() for o in self.ALLOWED_ORIGINS_RAW.split(",") if o]

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
    MODEL_PATH = os.getenv("MODEL_PATH", "data/random_forest_model.pkl")
    DEFAULT_SEASON_ID = int(os.getenv("DEFAULT_SEASON_ID", "0"))

settings = Settings()
