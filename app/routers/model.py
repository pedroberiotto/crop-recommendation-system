from fastapi import APIRouter, HTTPException
from app.model_store import model_store
from app.schemas import ModelInfoResponse, RollbackRequest, RollbackResponse
from crop_reco.logger import log_event
router = APIRouter()

@router.get('/info', response_model=ModelInfoResponse, summary='Active model metadata and available artifacts')
def model_info() -> ModelInfoResponse:
    info = model_store.info()
    return ModelInfoResponse(**info, available_artifacts=model_store.available_artifacts())

@router.post('/rollback', response_model=RollbackResponse, summary='Switch the active model to a previous version without restarting the container', responses={200: {'description': 'Rollback performed successfully'}, 404: {'description': 'Artifact not found'}, 400: {'description': 'Artifact exists but is not deserializable'}, 500: {'description': 'Internal error loading the artifact'}})
def rollback(body: RollbackRequest) -> RollbackResponse:
    try:
        info = model_store.rollback(body.artifact)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f'Invalid artifact: {exc}')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error loading artifact: {exc}')
    return RollbackResponse(status='ok', active_model=ModelInfoResponse(**info, available_artifacts=model_store.available_artifacts()))
