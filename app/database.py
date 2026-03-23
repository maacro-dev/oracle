from supabase import create_client, Client
from .config import settings

def get_supabase_client() -> Client:
    """Return a Supabase client instance."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE:
        raise ValueError("Supabase credentials not configured.")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE)
