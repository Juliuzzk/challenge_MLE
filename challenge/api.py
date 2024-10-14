from typing import List
import fastapi
import pandas as pd
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conlist, validator
from challenge.model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = "; ".join([err["msg"] for err in exc.errors()])
    return JSONResponse(status_code=400, content={"detail": error_messages})


# Modelo de vuelo
class Flight(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str

    # Validación para MES
    @validator("MES")
    def validate_mes(cls, v):
        if not (1 <= v <= 12):
            raise ValueError("MES must be between 1 and 12.")
        return v

    # Validación para TIPOVUELO
    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, v):
        if v not in ["N", "I"]:
            raise ValueError(
                "TIPOVUELO must be 'N' for national or 'I' for international."
            )
        return v


class FlightsRequest(BaseModel):
    flights: conlist(Flight, min_items=1)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


# Endpoint para predicción
@app.post("/predict", status_code=200)
async def post_predict(request: FlightsRequest) -> dict:
    try:
        df = pd.DataFrame([flight.dict() for flight in request.flights])

        features = model.preprocess(df)
        predictions = model.predict(features)

        return {"predict": predictions}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

