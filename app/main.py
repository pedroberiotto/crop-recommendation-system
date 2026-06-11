import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.model_store import model_store
from app.routers import health, model, predict
from crop_reco.logger import log_event, setup_logging
_MODEL_ARTIFACT = os.environ.get('MODEL_ARTIFACT', 'final_model.pkl')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log_event('api_startup', artifact=_MODEL_ARTIFACT)
    try:
        model_store.load(_MODEL_ARTIFACT)
        log_event('model_loaded_ok', artifact=_MODEL_ARTIFACT)
    except FileNotFoundError:
        logger.warning("Artifact '%s' not found. API in degraded mode. Run `make train` to generate the model.", _MODEL_ARTIFACT)
    except Exception as exc:
        logger.error('Failed to load model: %s', exc)
    yield
    log_event('api_shutdown')
app = FastAPI(title='Crop Recommendation API', description='Crop recommendation API based on soil and climate IoT sensor readings.', version='2.0.0', lifespan=lifespan)
app.include_router(health.router, tags=['Health'])
app.include_router(predict.router, tags=['Prediction'])
app.include_router(model.router, prefix='/model', tags=['Model'])
