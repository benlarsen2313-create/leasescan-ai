import os, io, json, re, asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import pdfplumber
from openai import OpenAI
import httpx
import stripe

# ââ Credentials ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
SUPABASE_URL     = os.environ.get("SUPABASE_URL", "https://zbsjbvaffkbwhliujqqy.supabase.co")
SUPABASE_ANON    = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE = os.environ.get("SUPABASE_SERVICE_KEY", "")
STRIPE_SECRET    = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID  = os.environ.get("STRIPE_PRICE_ID", "")
APP_URL          = os.environ.get("APP_URL", "https://www.leasescanai.com")
RENTCAST_API_KEY = os.environ.get("RENTCAST_API_KEY", "")

stripe.api_key = STRIPE_SECRET
client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# ââ Auth helper ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
async def get_user(request: Request) -> dict:
    """Verify Supabase JWT and return user dict, or raise 401."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = auth[7:]
    async with httpx.AsyncClient() as hc:
        r = await hc.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON,
            },
        )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return r.json()

async def require_active_subscription(user: dict = Depends(get_user)) -> dict:
    """Check that the user has an active Stripe subscription."""
    email = user.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Could not determine user email")

    # Look up Stripe customer by email
    customers = stripe.Customer.list(email=email, limit=1)
    if not customers.data:
        raise HTTPException(status_code=402, detail="No subscription found")

    customer_id = customers.data[0].id
    subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    return user

# ââ Stripe Checkout ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
@app.post("/api/create-checkout-session")
async def create_checkout_session(user: dict = Depends(get_user)):
    email = user.get("email", "")
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        customer_email=email,
        success_url=f"{APP_URL}/?subscribed=true",
        cancel_url=f"{APP_URL}/?canceled=true",
    )
    return {"url": session.url}

@app.get("/api/subscription-status")
async def subscription_status(user: dict = Depends(get_user)):
    email = user.get("email", "")
    customers = stripe.Customer.list(email=email, limit=1)
    if not customers.data:
        return {"active": False}
    customer_id = customers.data[0].id
    subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    return {"active": bool(subs.data)}

# ââ Lease analysis (requires active subscription) ââââââââââââââââââââââââââââââ
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    location: str = Form(""),
    bedrooms: int = Form(1),
    user: dict = Depends(require_active_subscription),
):
    content = await file.read()
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

    text = text[:15000]

    prompt = f"""You are an expert tenant-rights attorney and lease analyst.
Analyze the following residential lease agreement and return a JSON object with these exact keys:

- "overall_risk": one of "Low", "Medium", "High"
- "summary": 2-3 sentence plain-English overview
- "red_flags": array of objects with "title" and "description" â clauses that are risky, unusual, or tenant-unfavorable
- "good_clauses": array of objects with "title" and "description" â tenant-protective or fair clauses
- "missing_clauses": array of strings â important protections that are absent
- "market_comparison": 1-2 sentences comparing key terms to typical leases{' in ' + location if location else ''}
- "negotiation_tips": array of strings â actionable advice for negotiating better terms
- "key_dates": array of objects with "label" and "value" â important dates/deadlines
- "financial_summary": object with "monthly_rent", "security_deposit", "late_fee", "other_fees"

Lease text:
{text}

Return ONLY valid JSON, no markdown, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2500,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse analysis response.")

    # ── RentCast enrichment
    if RENTCAST_API_KEY:
        addr = data.get("street_address") or ""
        zip_c = data.get("zip_code") or ""
        if not zip_c:
            m = re.search(r"\\b(\\d{5})\\b", text)
            if m:
                zip_c = m.group(1)
        beds = data.get("bedrooms") or bedrooms or 1
        rent_data = await get_rent_data(addr, zip_c, beds)
        if rent_data:
            data["rent_data"] = rent_data
            lease_rent = data.get("monthly_rent")
            est = rent_data.get("estimate") or rent_data.get("avg_rent")
            if lease_rent and est:
                diff = lease_rent - est
                pct = round(abs(diff) / est * 100)
                if diff > 50:
                    data["rent_verdict"] = f"Your rent of ${lease_rent:,.0f}/mo is {pct}% above the market estimate of ${est:,.0f}/mo."
                elif diff < -50:
                    data["rent_verdict"] = f"Your rent of ${lease_rent:,.0f}/mo is {pct}% below the market estimate of ${est:,.0f}/mo — a good deal."
                else:
                    data["rent_verdict"] = f"Your rent of ${lease_rent:,.0f}/mo is in line with the market estimate of ${est:,.0f}/mo."

    return data

# ââ Health check âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ââ Serve frontend âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
@app.get("/")
def index():
    with open("index.html", "rb") as f:
        return Response(content=f.read(), media_type="text/html; charset=utf-8")

@app.get("/robots.txt")
def robots():
    with open("robots.txt") as f:
        return f.read()

@app.get("/sitemap.xml")
def sitemap():
    with open("sitemap.xml") as f:
        return f.read()
