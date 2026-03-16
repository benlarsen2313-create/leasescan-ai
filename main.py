"""
LeaseScan AI - Backend API
Analyzes real estate lease agreements for risks and compares rent to market rates.

SETUP: Copy .env.example to .env and fill in your API keys.
Users never see or need your keys — they just use the app and pay you.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import pdfplumber
import io
import json
import os
import httpx
from pathlib import Path

# Load API keys from .env file — users never see these
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
RENTCAST_API_KEY = os.getenv("RENTCAST_API_KEY", "")

if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set in .env file. See .env.example")


app = FastAPI(title="LeaseScan AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
frontend_path = Path(__file__).parent
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))

    @app.get("/robots.txt")
    async def robots():
        return FileResponse(str(frontend_path / "robots.txt"), media_type="text/plain")

    @app.get("/sitemap.xml")
    async def sitemap():
        return FileResponse(str(frontend_path / "sitemap.xml"), media_type="application/xml")


# ─── Prompts ──────────────────────────────────────────────────────────────────

LEASE_ANALYSIS_PROMPT = """You are an expert real estate attorney specializing in tenant protection.
Analyze the provided lease agreement THOROUGHLY and return ONLY valid JSON — no markdown, no preamble.

Return this exact JSON structure:
{
  "risk_score": <integer 0-100, where 0=perfectly safe, 100=extremely dangerous for tenant>,
  "risk_level": "<one of: Safe | Moderate | High Risk | Danger>",
  "summary": "<2-3 sentence plain-English overview of this lease>",
  "monthly_rent": <number or null>,
  "lease_term_months": <number or null>,
  "property_type": "<apartment|house|condo|studio|townhouse|other>",
  "property_address": "<full address found in lease, or null>",
  "overall_recommendation": "<one of: Sign As-Is | Negotiate First | Avoid if Possible>",
  "flagged_clauses": [
    {
      "id": <integer starting at 1>,
      "category": "<one of: Early Termination | Security Deposit | Rent Increases | Maintenance | Landlord Entry | Subletting | Auto-Renewal | Hidden Fees | Pet Policy | Move-Out Requirements | Utilities | Noise/Rules | Other>",
      "severity": "<high|medium|low>",
      "clause_text": "<exact quote from lease, max 200 chars>",
      "explanation": "<1-2 sentence plain-English explanation of why this is concerning>",
      "suggestion": "<specific negotiation advice or what to watch out for>"
    }
  ],
  "positive_clauses": [
    {
      "category": "<category>",
      "explanation": "<what this protects and why it's good for the tenant>"
    }
  ],
  "missing_standard_protections": [
    "<string describing a standard tenant protection NOT found in this lease>"
  ],
  "top_negotiation_points": [
    "<specific, actionable item to negotiate before signing — ordered most important first>"
  ]
}

Flag EVERY unusual, one-sided, or risky clause. Be thorough. If a clause is standard and fair, note it as positive.
Use plain English. Be specific with clause references when possible."""


RENT_ESTIMATE_PROMPT = """You are a real estate market analyst with knowledge of US rental markets.
Based on your training data, provide a rental market estimate for the given zip code.
Return ONLY valid JSON — no markdown, no explanation outside the JSON.

{
  "zip_code": "<zip provided>",
  "city_state": "<city, state for this zip>",
  "estimated_median_rent": <integer — estimated median monthly rent for a 1BR apartment in USD>,
  "studio_estimate": <integer or null>,
  "one_bed_estimate": <integer or null>,
  "two_bed_estimate": <integer or null>,
  "market_description": "<1-2 sentences describing this rental market>",
  "confidence": "<low|medium|high>",
  "disclaimer": "This is an AI estimate based on training data. Always verify with current listings."
}

Zip code: {zip_code}
Property type: {property_type}
Bedrooms (if known): {bedrooms}"""


# ─── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_lease(
    file: UploadFile = File(...),
    zip_code: str = Form(default=""),
    bedrooms: str = Form(default="unknown"),
):
    # API keys come from your .env — users never provide them
    openai_api_key = OPENAI_API_KEY
    rentcast_key = RENTCAST_API_KEY

    if not openai_api_key:
        raise HTTPException(status_code=500, detail="Server not configured. Please set OPENAI_API_KEY in your .env file.")
    # ── 1. Validate file ──────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # ── 2. Extract text from PDF ──────────────────────────────────────────────
    pdf_bytes = await file.read()
    lease_text = ""

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lease_text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {str(e)}")

    if len(lease_text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="PDF appears to be empty or is a scanned image (not machine-readable). Please use a text-based PDF.",
        )

    # ── 3. Analyze lease with OpenAI ──────────────────────────────────────────
    client = OpenAI(api_key=openai_api_key)

    try:
        lease_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": LEASE_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this lease agreement:\n\n{lease_text[:15000]}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4000,
        )
        analysis = json.loads(lease_response.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI analysis response.")
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {error_msg}")

    # ── 4. Get market rent data ───────────────────────────────────────────────
    market_data = None
    if zip_code:
        # Try RentCast first (live data)
        if rentcast_key:
            market_data = await get_rentcast_data(zip_code, rentcast_key)

        # Fall back to AI estimate
        if not market_data:
            try:
                rent_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a real estate market analyst. Return only valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": RENT_ESTIMATE_PROMPT.format(
                                zip_code=zip_code,
                                property_type=analysis.get("property_type", "apartment"),
                                bedrooms=bedrooms,
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                market_data = json.loads(rent_response.choices[0].message.content)
                market_data["source"] = "AI Estimate"
            except Exception:
                market_data = {
                    "source": "unavailable",
                    "zip_code": zip_code,
                    "error": "Could not retrieve market data.",
                }

    # ── 5. Build price verdict ────────────────────────────────────────────────
    price_verdict = None
    if (
        market_data
        and analysis.get("monthly_rent")
        and market_data.get("estimated_median_rent")
    ):
        lease_rent = float(analysis["monthly_rent"])
        market_median = float(market_data["estimated_median_rent"])
        diff_pct = ((lease_rent - market_median) / market_median) * 100

        if diff_pct > 15:
            price_verdict = {
                "status": "overpriced",
                "percent": round(abs(diff_pct)),
                "message": f"Rent is ~{round(abs(diff_pct))}% above market median",
                "detail": f"You're paying ${lease_rent:,.0f}/mo vs. ~${market_median:,.0f}/mo market average.",
            }
        elif diff_pct < -10:
            price_verdict = {
                "status": "deal",
                "percent": round(abs(diff_pct)),
                "message": f"Rent is ~{round(abs(diff_pct))}% below market — great deal!",
                "detail": f"You're paying ${lease_rent:,.0f}/mo vs. ~${market_median:,.0f}/mo market average.",
            }
        else:
            price_verdict = {
                "status": "fair",
                "percent": round(abs(diff_pct)),
                "message": "Rent is in line with market rates",
                "detail": f"You're paying ${lease_rent:,.0f}/mo vs. ~${market_median:,.0f}/mo market average.",
            }

    return {**analysis, "market_data": market_data, "price_verdict": price_verdict}


# ─── RentCast Integration ──────────────────────────────────────────────────────

async def get_rentcast_data(zip_code: str, api_key: str) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.rentcast.io/v1/markets",
                params={"zipCode": zip_code},
                headers={"X-Api-Key": api_key},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "RentCast (Live Data)",
                    "zip_code": zip_code,
                    "estimated_median_rent": data.get("averageRent"),
                    "one_bed_estimate": data.get("averageRent"),
                    "market_description": f"Live market data for {zip_code}.",
                    "confidence": "high",
                    "disclaimer": "Data provided by RentCast.",
                }
    except Exception:
        pass
    return None


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "LeaseScan AI"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
