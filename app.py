from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from predict_service import run_prediction

app = FastAPI(
    title="Weather Forecast API",
    description="7-day multi-task transformer weather forecast",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    print("")
    print("====================================================")
    print("        WEATHER SERVER STARTED SUCCESSFULLY")
    print("====================================================")
    print(" CLICK HERE TO OPEN THE DEMO: http://127.0.0.1:8000/")
    print("====================================================")
    print("")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    city: str
    end_date: Optional[str] = None
    time_option: Optional[str] = None
    timeframe: Optional[int] = None

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return run_prediction(
            city_code=req.city,
            time_option=req.time_option,
            timeframe_value=req.timeframe,
            end_date=req.end_date,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Server error: {ex}")

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)