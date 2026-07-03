"""
Umami ML Service - FastAPI Entry Point
GPU-accelerated ML inference for visitor journey analysis and recommendations.

Endpoints:
  POST /predict/next-page     - Next page prediction
  POST /predict/funnel-drop   - Funnel drop-off prediction
  POST /predict/intent        - Session intent classification
  POST /recommend             - Session-based recommendations
  GET  /health                - Health check + GPU status
  POST /train/next-page       - Train next page predictor
  POST /train/funnel          - Train funnel predictor
  POST /train/intent          - Train intent classifier
  POST /train/recommender     - Train recommender system
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from ml.config import CONFIG
from ml import gpu_utils
from ml.data.session_sequence import SessionDataExtractor

# Model instances (lazy-loaded)
_next_page = None
_funnel = None
_intent = None
_recommender = None
_rage_click = None
_journey_clusterer = None
_bandit = None
_ab_test = None

logger = logging.getLogger("umami-ml")


class NextPageRequest(BaseModel):
    website_id: str
    session_pages: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    use_transformer: bool = False


class FunnelRequest(BaseModel):
    website_id: str
    session: dict
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class IntentRequest(BaseModel):
    website_id: str
    session_pages: list[str] = Field(..., min_length=1)
    session_features: dict = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    website_id: str
    session_pages: list[str] = Field(default_factory=list)
    session_features: Optional[dict] = None
    top_k: int = Field(default=20, ge=1, le=100)


class TrainNextPageRequest(BaseModel):
    website_id: str
    start_date: str  # ISO format
    end_date: str
    min_session_length: int = Field(default=2, ge=1)
    use_transformer: bool = True
    epochs: int = Field(default=20, ge=1)


class TrainFunnelRequest(BaseModel):
    website_id: str
    start_date: str
    end_date: str
    funnel_window_minutes: int = Field(default=60, ge=1)
    funnel_steps: list[dict] = Field(default_factory=list)


class TrainIntentRequest(BaseModel):
    website_id: str
    start_date: str
    end_date: str


class TrainRecommenderRequest(BaseModel):
    website_id: str
    start_date: str
    end_date: str
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=256, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Log GPU status on startup
    logger.info(f"GPU Count: {gpu_utils.GPU_COUNT}")
    logger.info(f"Active Device: {gpu_utils.DEVICE}")
    logger.info(f"ONNX Providers: {gpu_utils.ONNX_PROVIDERS}")
    
    if gpu_utils.GPU_COUNT > 0:
        for i in range(gpu_utils.GPU_COUNT):
            info = gpu_utils.get_gpu_memory_info(i)
            logger.info(f"GPU {i}: {info['device']} - "
                       f"{info['free']/1e9:.1f}GB free / {info['total']/1e9:.1f}GB total")
    yield


app = FastAPI(
    title="Umami ML Service",
    description="GPU-accelerated ML for visitor journey analysis and recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_models():
    """Lazy-load models"""
    global _next_page, _funnel, _intent, _recommender
    
    if _next_page is None:
        from ml.models.next_page_predictor import NextPagePredictor
        _next_page = NextPagePredictor()
    
    if _funnel is None:
        from ml.models.funnel_predictor import FunnelPredictor
        _funnel = FunnelPredictor()
    
    if _intent is None:
        from ml.models.session_intent import SessionIntentClassifier
        _intent = SessionIntentClassifier()
    
    if _recommender is None:
        from ml.models.recommender import Recommender
        _recommender = Recommender()
    
    return _next_page, _funnel, _intent, _recommender

def get_rage_click():
    global _rage_click
    if _rage_click is None:
        from ml.models.rage_click import RageClickDetector
        _rage_click = RageClickDetector()
    return _rage_click

def get_journey_clusterer():
    global _journey_clusterer
    if _journey_clusterer is None:
        from ml.models.journey_clusterer import JourneyClusterer
        _journey_clusterer = JourneyClusterer()
    return _journey_clusterer

def get_bandit():
    global _bandit
    if _bandit is None:
        from ml.models.contextual_bandit import ContextualBandit
        _bandit = ContextualBandit()
    return _bandit

def get_ab_test():
    global _ab_test
    if _ab_test is None:
        from ml.models.ab_test import ABTestFramework
        _ab_test = ABTestFramework()
    return _ab_test


def get_extractor():
    return SessionDataExtractor(CONFIG.db.url)


# ============================================================
# Health & Status
# ============================================================

@app.get("/health")
async def health():
    """Health check with GPU status"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": {
            "count": gpu_utils.GPU_COUNT,
            "device": str(gpu_utils.DEVICE),
            "providers": gpu_utils.ONNX_PROVIDERS,
        },
    }
    
    if gpu_utils.GPU_COUNT > 0:
        status["gpu"]["memory"] = [
            gpu_utils.get_gpu_memory_info(i)
            for i in range(gpu_utils.GPU_COUNT)
        ]
    
    # Check if models are loaded
    np, fu, it, re = get_models()
    rc = get_rage_click()
    jc = get_journey_clusterer()
    bd = get_bandit()
    ab = get_ab_test()
    status["models"] = {
        "next_page_predictor": {"loaded": True, "trained": np.is_trained},
        "funnel_predictor": {"loaded": True, "trained": fu.is_trained},
        "session_intent": {"loaded": True, "trained": it.is_trained},
        "recommender": {"loaded": True, "trained": re.is_trained},
        "rage_click_detector": {"loaded": True, "trained": rc.is_trained},
        "journey_clusterer": {"loaded": True, "trained": jc.is_trained},
        "contextual_bandit": {"loaded": True, "trained": bd.is_trained},
        "ab_test_framework": {"loaded": True, "trained": ab.is_trained},
    }
    
    return status

@app.get("/")
async def root():
    return {"service": "Umami ML Service", "version": "1.0.0", "docs": "/docs"}


# ============================================================
# Prediction Endpoints
# ============================================================

@app.post("/predict/next-page")
async def predict_next_page(req: NextPageRequest):
    """Predict next page(s) a visitor will view"""
    np, _, _, _ = get_models()
    
    if not np.is_trained:
        raise HTTPException(400, "Next page predictor not trained. Run /train/next-page first.")
    
    try:
        results = np.predict(req.session_pages, req.top_k, req.use_transformer)
        return {
            "website_id": req.website_id,
            "session_pages": req.session_pages,
            "predictions": results,
            "model_type": "transformer" if req.use_transformer and np.transformer else "markov",
            "device": str(gpu_utils.DEVICE),
        }
    except Exception as e:
        logger.error(f"Next page prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/funnel-drop")
async def predict_funnel_drop(req: FunnelRequest):
    """Predict if a visitor will drop off at current funnel step"""
    _, fu, _, _ = get_models()
    
    if not fu.is_trained:
        raise HTTPException(400, "Funnel predictor not trained. Run /train/funnel first.")
    
    try:
        result = fu.predict(req.session, req.threshold)
        result["website_id"] = req.website_id
        return result
    except Exception as e:
        logger.error(f"Funnel prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/intent")
async def predict_intent(req: IntentRequest):
    """Classify visitor session intent"""
    _, _, it, _ = get_models()
    
    if not it.is_trained:
        raise HTTPException(400, "Intent classifier not trained. Run /train/intent first.")
    
    try:
        result = it.predict(req.session_pages, req.session_features)
        result["website_id"] = req.website_id
        return result
    except Exception as e:
        logger.error(f"Intent prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """Get content recommendations for a session"""
    _, _, _, re = get_models()
    
    if not re.is_trained:
        raise HTTPException(400, "Recommender not trained. Run /train/recommender first.")
    
    try:
        results = re.recommend(req.session_pages, req.session_features, req.top_k)
        return {
            "website_id": req.website_id,
            "session_pages": req.session_pages,
            "recommendations": results,
            "count": len(results),
            "is_cold_start": len(req.session_pages) < 2,
        }
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(500, str(e))


class RageClickRequest(BaseModel):
    website_id: str
    session_id: Optional[str] = None
    clicks: Optional[list[dict]] = None


@app.post("/predict/rage-click")
async def predict_rage_click(req: RageClickRequest):
    """Detect user frustration signals from click/heatmap data"""
    rc = get_rage_click()

    if req.session_id:
        result = rc.predict_session(req.session_id, req.website_id)
    elif req.clicks:
        result = rc.predict(req.clicks)
    else:
        raise HTTPException(400, "Provide session_id or clicks array")

    return {"website_id": req.website_id, **result}


class ClusterRequest(BaseModel):
    website_id: str
    session: dict


@app.post("/predict/cluster")
async def predict_cluster(req: ClusterRequest):
    """Predict visitor archetype for a session"""
    jc = get_journey_clusterer()
    result = jc.predict(req.session)
    return {"website_id": req.website_id, **result}


class BanditRecommendRequest(BaseModel):
    website_id: str
    session: dict
    available_pages: list[str]
    top_k: int = 5


@app.post("/recommend/bandit")
async def bandit_recommend(req: BanditRecommendRequest):
    """Contextual bandit recommendations"""
    bd = get_bandit()
    if not bd.is_trained:
        bd.train()
    results = bd.recommend(req.session, req.available_pages, req.top_k)
    return {
        "website_id": req.website_id,
        "recommendations": results,
        "count": len(results),
    }


class BanditRewardRequest(BaseModel):
    website_id: str
    page: str
    session: dict
    reward: float


@app.post("/recommend/bandit/reward")
async def bandit_reward(req: BanditRewardRequest):
    """Record reward for a bandit recommendation"""
    bd = get_bandit()
    bd.record_reward(req.page, req.session, req.reward)
    return {"status": "recorded"}


class ABTestCreateRequest(BaseModel):
    experiment_id: str
    name: str
    variants: list[dict]
    traffic_fraction: float = 1.0
    min_sample_size: int = 100


class ABTestAssignRequest(BaseModel):
    experiment_id: str
    visitor_id: str


class ABTestRecordRequest(BaseModel):
    experiment_id: str
    variant_id: str
    converted: bool


@app.post("/ab-test/create")
async def ab_test_create(req: ABTestCreateRequest):
    """Create an A/B test experiment"""
    ab = get_ab_test()
    result = ab.create_experiment(
        req.experiment_id, req.name, req.variants,
        req.traffic_fraction, req.min_sample_size,
    )
    return {"status": "created", **result}


@app.post("/ab-test/assign")
async def ab_test_assign(req: ABTestAssignRequest):
    """Assign a visitor to a variant"""
    ab = get_ab_test()
    variant = ab.assign(req.experiment_id, req.visitor_id)
    return {"experiment_id": req.experiment_id, "variant_id": variant}


@app.post("/ab-test/record")
async def ab_test_record(req: ABTestRecordRequest):
    """Record a conversion for a variant"""
    ab = get_ab_test()
    ab.record(req.experiment_id, req.variant_id, req.converted)
    return {"status": "recorded"}


@app.get("/ab-test/results/{experiment_id}")
async def ab_test_results(experiment_id: str):
    """Get experiment results with statistical significance"""
    ab = get_ab_test()
    return ab.get_results(experiment_id)


@app.get("/ab-test/results")
async def ab_test_all_results():
    """Get all experiment results"""
    ab = get_ab_test()
    return ab.get_results()


# ============================================================
# Training Endpoints
# ============================================================

@app.post("/train/next-page")
async def train_next_page(req: TrainNextPageRequest):
    """Train the next page predictor from session data"""
    np, _, _, _ = get_models()
    
    try:
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date)
        
        with get_extractor() as extractor:
            logger.info(f"Extracting sessions for {req.website_id}...")
            sequences = extractor.get_session_sequences(
                req.website_id, start, end, req.min_session_length
            )
            
            page_sequences = [s.pages for s in sequences if len(s.pages) >= req.min_session_length]
            
            logger.info(f"Training on {len(page_sequences)} sequences...")
            np.train(
                page_sequences,
                use_transformer=req.use_transformer,
                epochs=req.epochs,
            )
        
        return {
            "status": "complete",
            "model": "next_page_predictor",
            "sequences_trained": len(page_sequences),
            "vocab_size": np.vocab_size,
            "vocabulary": list(np.page_to_idx.keys())[:20],  # preview
            "device": str(gpu_utils.DEVICE),
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/train/funnel")
async def train_funnel(req: TrainFunnelRequest):
    """Train funnel drop-off predictor"""
    _, fu, _, _ = get_models()
    
    try:
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date)
        
        with get_extractor() as extractor:
            sessions_df = extractor.get_session_features(req.website_id, start, end)
            
            if sessions_df.empty:
                raise HTTPException(400, "No session data found")
            
            # Prepare training data
            sessions = sessions_df.to_dict('records')
            
            # Create labels: 1 = completed funnel (had event_name or specific path)
            # Simple heuristic: sessions with multiple pageviews are "converted"
            labels = [
                1 if s.get('total_pageviews', 0) > 3 else 0
                for s in sessions
            ]
            
            fu.train(sessions, labels)
        
        return {
            "status": "complete",
            "model": "funnel_predictor",
            "sessions_trained": len(sessions),
            "train_accuracy": fu.metadata.get('train_accuracy'),
            "val_accuracy": fu.metadata.get('val_accuracy'),
        }
    except Exception as e:
        logger.error(f"Funnel training failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/train/intent")
async def train_intent(req: TrainIntentRequest):
    """Train session intent classifier"""
    _, _, it, _ = get_models()
    
    try:
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date)
        
        with get_extractor() as extractor:
            sequences = extractor.get_session_sequences(req.website_id, start, end)
            
            if not sequences:
                raise HTTPException(400, "No session data found")
            
            sessions_df = extractor.get_session_features(req.website_id, start, end)
            session_dict = {s['session_id']: s for s in sessions_df.to_dict('records')}
            
            # Build training data
            page_sequences = []
            session_features = []
            labels = []
            
            for seq in sequences:
                if len(seq.pages) < 2:
                    continue
                page_sequences.append(seq.pages)
                
                sf = session_dict.get(seq.session_id, {})
                sf['page_views'] = len(seq.pages)
                sf['session_duration'] = seq.duration_seconds
                session_features.append(sf)
                
                # Simple label heuristic based on behavior
                visited_checkout = any('checkout' in p.lower() or 'cart' in p.lower() for p in seq.pages)
                visited_pricing = any('pricing' in p.lower() or 'price' in p.lower() for p in seq.pages)
                visited_support = any('support' in p.lower() or 'help' in p.lower() or 'contact' in p.lower() for p in seq.pages)
                
                if visited_checkout:
                    labels.append('ready_to_buy')
                elif visited_pricing:
                    labels.append('price_comparing')
                elif visited_support:
                    labels.append('support_seeking')
                elif seq.duration_seconds > 120:
                    labels.append('researching')
                elif len(seq.pages) <= 2:
                    labels.append('just_browsing')
                else:
                    labels.append('content_consumption')
            
            it.train(session_features, page_sequences, labels)
        
        return {
            "status": "complete",
            "model": "session_intent",
            "sessions_trained": len(page_sequences),
            "train_accuracy": it.metadata.get('train_accuracy'),
        }
    except Exception as e:
        logger.error(f"Intent training failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/train/recommender")
async def train_recommender(req: TrainRecommenderRequest):
    """Train session-based recommender system"""
    _, _, _, re = get_models()
    
    try:
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date)
        
        with get_extractor() as extractor:
            sequences = extractor.get_session_sequences(req.website_id, start, end)
            
            if not sequences:
                raise HTTPException(400, "No session data found")
            
            page_sequences = [s.pages for s in sequences if len(s.pages) >= 3]
            
            if len(page_sequences) < 10:
                raise HTTPException(400, f"Need at least 10 sessions (got {len(page_sequences)}")
            
            re.train_session_encoder(page_sequences, epochs=req.epochs, batch_size=req.batch_size)
        
        return {
            "status": "complete",
            "model": "recommender",
            "sessions_trained": len(page_sequences),
            "vocab_size": re.vocab_size,
            "pages_indexed": len(re.page_embeddings),
        }
    except Exception as e:
        logger.error(f"Recommender training failed: {e}")
        raise HTTPException(500, str(e))


class TrainRageClickRequest(BaseModel):
    website_id: str
    start_date: str
    end_date: str


@app.post("/train/rage-click")
async def train_rage_click(req: TrainRageClickRequest):
    """Train rage click detector from heatmap data"""
    rc = get_rage_click()

    try:
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date)

        with get_extractor() as extractor:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT session_id, visit_id,
                       array_agg(ROW(x, y, page_x, page_y, scroll_pct,
                               viewport_w, viewport_h, page_h, created_at)
                               ORDER BY created_at) AS clicks
                FROM heatmap_event
                WHERE website_id = %s
                  AND created_at BETWEEN %s AND %s
                GROUP BY session_id, visit_id
                HAVING COUNT(*) >= 2
                LIMIT 5000
            """, (req.website_id, start, end))

            click_sessions = []
            for row in cur.fetchall():
                clicks_raw = row['clicks']
                clicks = [
                    {
                        'x': c[0], 'y': c[1], 'page_x': c[2], 'page_y': c[3],
                        'scroll_pct': c[4], 'viewport_w': c[5], 'viewport_h': c[6],
                        'page_h': c[7], 'created_at': c[8],
                    }
                    for c in clicks_raw
                ]
                features = rc.extract_click_features(clicks)
                if features:
                    click_sessions.append(features)

            cur.close()
            conn.close()

            rc.train(click_sessions)

        return {
            "status": "complete",
            "model": "rage_click_detector",
            "sessions_trained": len(click_sessions),
        }
    except Exception as e:
        logger.error(f"Rage click training failed: {e}")
        raise HTTPException(500, str(e))


# ============================================================
# Model Management
# ============================================================

@app.post("/models/save")
async def save_models():
    """Save all trained models to disk"""
    np, fu, it, re = get_models()
    saved = []
    
    if np.is_trained:
        saved.append(np.save())
    if fu.is_trained:
        saved.append(fu.save())
    if it.is_trained:
        saved.append(it.save())
    if re.is_trained:
        saved.append(re.save())
    
    return {"status": "complete", "saved": saved}


@app.post("/models/load")
async def load_models():
    """Load all saved models from disk"""
    np, fu, it, re = get_models()
    loaded = []
    
    try:
        np.load()
        loaded.append("next_page_predictor")
    except FileNotFoundError:
        pass
    
    try:
        fu.load()
        loaded.append("funnel_predictor")
    except FileNotFoundError:
        pass
    
    try:
        it.load()
        loaded.append("session_intent")
    except FileNotFoundError:
        pass
    
    try:
        re.load()
        loaded.append("recommender")
    except FileNotFoundError:
        pass
    
    return {"status": "complete", "loaded": loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=CONFIG.api.host,
        port=CONFIG.api.port,
        log_level=CONFIG.api.log_level,
        reload=CONFIG.api.auto_reload,
    )
