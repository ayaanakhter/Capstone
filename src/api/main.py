import sys
import os
import asyncio
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detector.model import LieDetectorModel
from src.api.database import init_db, get_db, Session as SessionModel

app = FastAPI(title="Hallucination Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB tables on startup
init_db()

detector_components = {}

def get_detector():
    if "model" not in detector_components:
        print("Loading Qwen model...")
        detector_components["model"] = LieDetectorModel(model_name="./QWEN")
        print("Model loaded.")
    return detector_components


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate_stream(req: GenerateRequest, db: DBSession = Depends(get_db)):
    """
    True Real-Time Streaming — yields tokens immediately via SSE.
    Saves the completed session to SQLite when generation finishes.
    """
    components = get_detector()
    model = components["model"]

    print(f"Streaming generation for prompt: '{req.prompt}'")

    token_generator = model.generate_stream(
        req.prompt,
        max_new_tokens=req.max_tokens,
        compute_grads=True,
        mc_dropout_passes=2
    )

    async def event_generator():
        yield f"data: {json.dumps({'event': 'start'})}\n\n"

        full_text = ""
        all_lie_scores = []
        all_entropy = []
        all_grad_norm = []
        all_mc_var = []
        flagged_count = 0

        for sig in token_generator:
            # Normalize signals
            norm_entropy  = min(sig.entropy / 10.0, 1.0)
            norm_grad     = min(sig.gradient_norm / 10.0, 1.0)
            norm_mc       = min(sig.mc_variance / 1.0, 1.0)

            raw_uncertainty = (0.15 * norm_entropy) + (0.30 * norm_grad) + (0.25 * norm_mc)
            lie_product = sig.confidence * raw_uncertainty
            x = 5.0 * (lie_product - 0.3)
            lie_score = float(1 / (1 + np.exp(-x)))

            is_flagged = lie_score > 0.5
            if is_flagged:
                flagged_count += 1

            all_lie_scores.append(lie_score)
            all_entropy.append(norm_entropy)
            all_grad_norm.append(norm_grad)
            all_mc_var.append(norm_mc)
            full_text += sig.token

            mean_lie = sum(all_lie_scores) / len(all_lie_scores)

            payload = {
                "event":             "token",
                "token":             sig.token,
                "confidence":        float(sig.confidence),
                "entropy":           float(norm_entropy),
                "gradient_norm":     float(norm_grad),
                "mc_variance":       float(norm_mc),
                "lie_score":         lie_score,
                "is_flagged":        bool(is_flagged),
                "overall_lie_score": float(mean_lie),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.01)

        # ── Save completed session to SQLite ──────────────────────────────
        if all_lie_scores:
            try:
                record = SessionModel(
                    prompt=req.prompt,
                    response=full_text.strip(),
                    hallucination_risk=float(np.mean(all_lie_scores)),
                    avg_entropy=float(np.mean(all_entropy)),
                    avg_gradient_norm=float(np.mean(all_grad_norm)),
                    avg_mc_variance=float(np.mean(all_mc_var)),
                    total_tokens=len(all_lie_scores),
                    flagged_tokens=flagged_count,
                )
                db.add(record)
                db.commit()
                print(f"Session saved to DB (id={record.id})")
            except Exception as e:
                print(f"DB save failed: {e}")

        yield f"data: {json.dumps({'event': 'end', 'full_text': full_text})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history")
def get_history(limit: int = 50, db: DBSession = Depends(get_db)):
    """Return last N sessions from the database, newest first."""
    rows = (
        db.query(SessionModel)
        .order_by(SessionModel.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":                row.id,
            "prompt":            row.prompt,
            "response":          row.response,
            "hallucination_risk": round(row.hallucination_risk, 3),
            "avg_entropy":       round(row.avg_entropy, 3),
            "avg_gradient_norm": round(row.avg_gradient_norm, 3),
            "avg_mc_variance":   round(row.avg_mc_variance, 3),
            "total_tokens":      row.total_tokens,
            "flagged_tokens":    row.flagged_tokens,
            "created_at":        row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@app.delete("/history/{session_id}")
def delete_session(session_id: int, db: DBSession = Depends(get_db)):
    """Delete a single session by ID."""
    row = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not row:
        return {"error": "Not found"}
    db.delete(row)
    db.commit()
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
