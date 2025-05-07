import traceback

import asyncio

from contextlib import asynccontextmanager

import uvicorn

from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from model import Model
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('Using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the model
    try:
        model_wrapper = Model(model_path=CONFIG['MODEL_PATH'])
        app.state.model = model_wrapper
    except Exception as e:
        logger.exception("Failed to initialize model during startup")
        app.state.model = None
    yield

# Initialize API Server
app = FastAPI(
    title="BERT Model",
    description="A BERT model for predicting review ratings",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
    lifespan=lifespan
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

@app.get("/")
def read_root():
    return {"message": "Welcome to the API."}

@app.get("/health", response_model=PredictionResponse | ErrorResponse)
async def get_health(request: Request):
    """
    Health check endpoint to verify if the model is loaded and ready for predictions.
    """
    try:
        model_wrapper = request.app.state.model

        if model_wrapper.session is None:
            raise RuntimeError("ONNX model session not initialized")
        
        return JSONResponse(status_code=200, content={"rating": 0.0})  # Dummy payload for health check
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(e),
                "traceback": traceback.format_exc() if CONFIG["DEBUG"] else None
            }
        )
    
@app.exception_handler(FileNotFoundError)
async def file_not_found_error_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": True, "message": f"File not found: {exc.filename}"},
    )

@app.exception_handler(asyncio.TimeoutError)
async def timeout_error_handler(request: Request, exc: asyncio.TimeoutError):
    return JSONResponse(
        status_code=504,
        content={"error": True, "message": "Request timed out"},
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": str(exc)},
    )


@app.post("/predict", response_model=PredictionResponse | ErrorResponse)
async def predict(request: Request, payload: ReviewRequest):
    """
    Predict the rating of a review text using the BERT model.
    """

    try:

        model_wrapper = request.app.state.model
        prediction = model_wrapper.predict(payload.reviewText)
        return {"rating": int(round(prediction))}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(e),
                "traceback": traceback.format_exc() if CONFIG["DEBUG"] else None
            }
        )


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, debug=True, log_config="log.ini"
                )