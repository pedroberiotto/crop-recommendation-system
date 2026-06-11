from __future__ import annotations
import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from app.model_store import model_store
from app.schemas import Alternative, Explanation, Prediction, PredictRequest, PredictRequestLax, PredictResponse
from crop_reco.config import FEATURE_BOUNDS, NUMERIC_FEATURES
from crop_reco.explainability import explain_predictions
from crop_reco.logger import log_request
router = APIRouter()

def _detect_clipping(df: pd.DataFrame) -> bool:
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            s = df[col]
            if (s < lo).any() or (s > hi).any():
                return True
    return False

@router.post('/predict', response_model=PredictResponse, summary='Recommend crop(s) from sensor readings', responses={503: {'description': 'Model not loaded'}, 422: {'description': 'Invalid payload (missing field, wrong type, or — in strict mode — outside the physical domain)'}})
async def predict(request: Request, explain: bool=Query(default=True, description='Include SHAP explanation for predictions with confidence > 0.70. Pass ?explain=false to disable and reduce latency for large batches.'), strict: bool=Query(default=True, description='strict=true (default): values outside the physical domain return 422. strict=false: values are clipped by PhysicalBoundsClipper and the request is logged with input_modified=true.')) -> PredictResponse:
    if not model_store.is_loaded:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    raw_body = await request.json()
    try:
        if strict:
            body = PredictRequest.model_validate(raw_body)
        else:
            body = PredictRequestLax.model_validate(raw_body)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={'detail': exc.errors()})
    t0 = time.perf_counter()
    records = [r.model_dump() for r in body.records]
    df = pd.DataFrame(records)[NUMERIC_FEATURES]
    input_modified = _detect_clipping(df)
    proba = model_store.predict_proba(df)
    classes = model_store.classes()
    active_model = model_store.get_model()
    sorted_indices = [np.argsort(row)[::-1] for row in proba]
    predicted_crops = [classes[idxs[0]] for idxs in sorted_indices]
    confidences = [float(row[idxs[0]]) for row, idxs in zip(proba, sorted_indices)]
    explanations_by_index: dict[int, list[dict] | None] = {}
    if explain:
        explain_indices = [i for i, confidence in enumerate(confidences) if confidence > 0.7]
        if explain_indices:
            raw_explanations = explain_predictions(active_model, df.iloc[explain_indices].reset_index(drop=True), [predicted_crops[i] for i in explain_indices], classes)
            explanations_by_index = dict(zip(explain_indices, raw_explanations))
    predictions: list[Prediction] = []
    for i, (probs, top_idx, crop, confidence) in enumerate(zip(proba, sorted_indices, predicted_crops, confidences)):
        alternatives = [Alternative(crop=classes[j], probability=round(float(probs[j]), 6)) for j in top_idx[1:4]]
        raw_explanation = explanations_by_index.get(i)
        shap_explanation = [Explanation(**item) for item in raw_explanation] if raw_explanation else None
        predictions.append(Prediction(crop=crop, confidence=round(confidence, 6), alternatives=alternatives, explanation=shap_explanation))
    latency_ms = (time.perf_counter() - t0) * 1000
    log_request(endpoint='/predict', latency_ms=latency_ms, batch_size=len(records), model_artifact=model_store.info().get('artifact'), input_modified=input_modified)
    return PredictResponse(predictions=predictions)
