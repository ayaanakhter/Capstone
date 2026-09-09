import sys
import os
import time
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import numpy as np

# Add parent dir to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detector.model import LieDetectorModel

app = FastAPI(title="The Lie Detector API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector_components = {}

def get_detector():
    if "model" not in detector_components:
        print("Loading model in bfloat16 for streaming...")
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
async def generate_stream(req: GenerateRequest):
    """
    True Real-Time Streaming: Yields tokens the millisecond they are generated.
    Memory optimized for CPU by flushing graphs instantly.
    """
    components = get_detector()
    model = components["model"]

    print(f"Streaming generation for prompt: '{req.prompt}'")

    # The Generator from model.py
    token_generator = model.generate_stream(
        req.prompt, 
        max_new_tokens=req.max_tokens, 
        compute_grads=True, 
        mc_dropout_passes=2
    )

    async def event_generator():
        yield f"data: {json.dumps({'event': 'start'})}\n\n"
        
        full_text = ""
        overall_lie_scores = []

        for sig in token_generator:
            # Calculate Lie Score mathematically for this exact token
            # normalized roughly (max entropy ~10, max grad_norm ~10, max var ~1)
            norm_entropy = min(sig.entropy / 10.0, 1.0)
            norm_grad = min(sig.gradient_norm / 10.0, 1.0)
            norm_mc = min(sig.mc_variance / 1.0, 1.0)
            
            raw_uncertainty = (0.15 * norm_entropy) + (0.30 * norm_grad) + (0.25 * norm_mc)
            lie_product = sig.confidence * raw_uncertainty
            
            # Sigmoid normalization
            alpha = 5.0
            beta = 0.3
            x = alpha * (lie_product - beta)
            lie_score = 1 / (1 + np.exp(-x))
            
            overall_lie_scores.append(lie_score)
            mean_lie_score = sum(overall_lie_scores) / len(overall_lie_scores)
            
            is_flagged = lie_score > 0.5
            full_text += sig.token

            payload = {
                "event": "token",
                "token": sig.token.replace('', ''),
                "confidence": float(sig.confidence),
                "entropy": float(norm_entropy),
                "gradient_norm": float(norm_grad),
                "mc_variance": float(norm_mc),
                "lie_score": float(lie_score),
                "is_flagged": bool(is_flagged),
                "overall_lie_score": float(mean_lie_score)
            }
            
            yield f"data: {json.dumps(payload)}\n\n"
            
            # Tiny sleep to allow asyncio to flush the stream to the client
            await asyncio.sleep(0.01)
            
        yield f"data: {json.dumps({'event': 'end', 'full_text': full_text})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
