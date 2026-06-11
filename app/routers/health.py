from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.model_store import model_store
from app.schemas import HealthResponse
router = APIRouter()

@router.get('/health', summary='API status and model availability', responses={200: {'description': 'API operational with model loaded'}, 503: {'description': 'API operational but model not loaded'}})
def health():
    loaded = model_store.is_loaded
    body = HealthResponse(status='ok' if loaded else 'degraded', model_loaded=loaded)
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())
