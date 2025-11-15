from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends,BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer,OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from passlib.context import CryptContext
from fastapi.openapi.utils import get_openapi
import jwt
import pytesseract
import hashlib
from PIL import Image
import io
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from typing import Optional
from bson import ObjectId
# -------------------- LOAD CONFIG --------------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
bearer_scheme = HTTPBearer()
# -------------------- FASTAPI INIT --------------------
app = FastAPI(title="KYC Verification API")

# CORS: allow your frontend origins (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="KYC Verification API",
        version="1.0.0",
        description="Backend for AI-Powered Identity Verification and Fraud Detection",
        routes=app.routes,
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    openapi_schema["components"]["securitySchemes"]["bearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }
    openapi_schema["security"] = [{"bearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi

from app.db import (
    users_coll,
    documents_coll,
    kyc_coll,
    verification_logs_coll,
    fraud_alerts_coll
)
# -------------------- SECURITY --------------------
# Use a PBKDF2-based scheme to avoid optional native bcrypt dependency issues.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password: str):
    # Use the configured pwd_context to hash the password.
    # PBKDF2 doesn't have the 72-byte bcrypt limit, so we can hash directly.
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])        
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        user = await users_coll.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------- VALIDATORS --------------------
def is_valid_email(email: str) -> bool:
    return re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email) is not None

def is_valid_password(password: str) -> bool:
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{8,16}$"
    return re.match(pattern, password) is not None

# -------------------- AUTH ROUTES --------------------
@app.post("/signup", tags=["Authentication"])
async def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):

    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    if not is_valid_password(password):
        raise HTTPException(status_code=400, detail="Password must meet rules")

    existing = await users_coll.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    await users_coll.insert_one({
        "name": name,
        "email": email,
        "password": hash_password(password),
        "createdAt": datetime.utcnow()
    })
    return {"message": "Signup successful"}

@app.post("/login", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_coll.find_one({"email": form_data.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email")

    if not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

# -------------------- OCR PARSER --------------------
def parse_text(text: str):
    parsed = {
        "panNumber": None,
        "aadhaarNumber": None,
        "name": None,
        "fatherName": None,
        "dob": None,
        "gender": None,
        "address": None,
    }
    # Clean text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_text = " ".join(lines)
    # --- Aadhaar detection ---
    aadhaar_match = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", full_text)
    if aadhaar_match:
        parsed["aadhaarNumber"] = aadhaar_match.group().replace(" ", "")
    # --- PAN detection ---
    pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", full_text)
    if pan_match:
        parsed["panNumber"] = pan_match.group()
    # --- DOB detection ---
    dob_match = re.search(r"\b(\d{2,4}[-/]\d{2}[-/]\d{2,4})\b", full_text)
    if dob_match:
        parsed["dob"] = dob_match.group(1)
    # --- Gender detection ---
    gender_match = re.search(r"(?i)\b(male|female|transgender|m|f)\b", full_text)
    if gender_match:
        g = gender_match.group(1).lower()
        parsed["gender"] = "Male" if g in ["m", "male"] else "Female" if g in ["f", "female"] else "Transgender"
    # --- Name detection (for both PAN & Aadhaar) ---
    name_match = re.search(r"(?i)\bname[:\-]?\s*([A-Za-z\s]+)", full_text)
    if name_match:
        name = name_match.group(1).strip()
        # Clean out noise (like DOB, Gender, Address words)
        name = re.split(r"\b(dob|date|gender|address|s/d|w/o|father)\b", name, flags=re.I)[0].strip()
        parsed["name"] = name
    # --- Father's name (specific for PAN) ---
    father_match = re.search(r"(?i)Father'?s Name[:\-]?\s*([A-Za-z\s]+?)(?=\s*(photo|signature|$))", full_text)
    if father_match:
        parsed["fatherName"] = father_match.group(1).strip()
    # --- Address detection (specific for Aadhaar) ---
    addr_match = re.search(r"(?i)address[:\-]?\s*(.+)", full_text)
    if addr_match:
        addr = addr_match.group(1).strip()
        # Stop if Aadhaar number or DOB appears afterward
        addr = re.split(r"\b(\d{4}\s?\d{4}\s?\d{4}|dob|date|gender)\b", addr, flags=re.I)[0].strip()
        parsed["address"] = addr
    else:
        # Fallback: try to detect line after "Address" keyword
        for i, line in enumerate(lines):
            if "address" in line.lower() and i + 1 < len(lines):
                parsed["address"] = lines[i + 1]
                break
    return parsed
# ---------- Milestone2: Verification helpers & endpoint ----------
def verify_aadhaar_format(aadhaar_number: str) -> dict:
    """
    Basic Aadhaar format check:
    - 12 digits, not starting with 0 or 1 (UIDAI numbers typically start with 2-9).
    Returns dict { valid: bool, message: str }.
    """
    if not aadhaar_number:
        return {"valid": False, "message": "No Aadhaar number provided"}
    normalized = re.sub(r"\s+", "", aadhaar_number)
    if re.fullmatch(r"^[2-9][0-9]{11}$", normalized):
        return {"valid": True, "message": "Aadhaar format valid"}
    return {"valid": False, "message": "Invalid Aadhaar format"}

def verify_pan_format(pan_number: str) -> dict:
    """
    Basic PAN format check:
    - 5 letters, 4 digits, 1 letter e.g. ABCDE1234F
    """
    if not pan_number:
        return {"valid": False, "message": "No PAN number provided"}
    normalized = pan_number.replace(" ", "").upper()
    if re.fullmatch(r"^[A-Z]{5}[0-9]{4}[A-Z]$", normalized):
        return {"valid": True, "message": "PAN format valid"}
    return {"valid": False, "message": "Invalid PAN format"}

@app.post("/verify-doc", tags=["Fraud Detection / Verification"])
async def verify_document(doc_id: str | None = None, current_user: dict = Depends(get_current_user)):

    query = {"userId": str(current_user["_id"])}

    if doc_id:
        try:
            query["_id"] = ObjectId(doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")

        doc = await documents_coll.find_one(query)
    else:
        doc = await documents_coll.find_one(query, sort=[("uploadedAt", -1)])

    if not doc:
        raise HTTPException(status_code=404, detail="No document found for this user")

    parsed = doc.get("parsed", {})

    result = {
        "documentId": str(doc.get("_id")),
        "filename": doc.get("filename"),
        "docType": doc.get("docType", "UNKNOWN"),
        "verification": {}
    }

    # Aadhaar
    aadhaar = parsed.get("aadhaarNumber")
    pan = parsed.get("panNumber")

    if aadhaar:
        v = verify_aadhaar_format(aadhaar)
        result["verification"]["aadhaar"] = {"value": aadhaar, **v}
    else:
        result["verification"]["aadhaar"] = {"value": None, "valid": False, "message": "No Aadhaar number found"}

    if pan:
        p = verify_pan_format(pan)
        result["verification"]["pan"] = {"value": pan, **p}
    else:
        result["verification"]["pan"] = {"value": None, "valid": False, "message": "No PAN number found"}

    overall_valid = (
        result["verification"]["aadhaar"]["valid"] or 
        result["verification"]["pan"]["valid"]
    )

    result["verification"]["overall"] = {
        "valid": overall_valid,
        "message": "At least one ID format valid" if overall_valid else "No valid Aadhaar/PAN format found"
    }

    await documents_coll.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "verificationResult": result["verification"],
                "verifiedAt": datetime.utcnow().isoformat()
            }
        }
    )
    return JSONResponse(content={"message": "Verification complete", "data": result})

def hash_identifier(identifier: str) -> str:
    """Hash Aadhaar or PAN using SHA256 for privacy."""
    if not identifier:
        return None
    return hashlib.sha256(identifier.encode()).hexdigest()

@app.post("/fraud-check", tags=["Fraud Detection / Verification"])
async def fraud_check(current_user: dict = Depends(get_current_user)):
    """
    Detect duplicate Aadhaar/PAN usage across users.
    Returns a simple fraud risk level.
    """

    cursor = documents_coll.find({"userId": str(current_user["_id"])})
    docs = [doc async for doc in cursor]

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for user")

    result = {
        "user": current_user["email"],
        "duplicates": [],
        "riskScore": 0,
        "riskLevel": "Low"
    }

    seen_hashes = set()

    # ---- async loop ----
    for doc in docs:
        parsed = doc.get("parsed", {})
        aadhaar = parsed.get("aadhaarNumber")
        pan = parsed.get("panNumber")

        # --- Aadhaar duplicate check ---
        if aadhaar:
            aadhaar_hash = hash_identifier(aadhaar)
            seen_hashes.add(aadhaar_hash)

            dup_aadhaar = await documents_coll.find_one({
                "userId": {"$ne": str(current_user["_id"])},
                "aadhaarHash": aadhaar_hash
            })

            if dup_aadhaar:
                result["duplicates"].append({
                    "type": "Aadhaar",
                    "value": aadhaar,
                    "conflictWithUser": dup_aadhaar.get("userId")
                })
                result["riskScore"] += 40

            await documents_coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"aadhaarHash": aadhaar_hash}}
            )

        # --- PAN duplicate check ---
        if pan:
            pan_hash = hash_identifier(pan)
            seen_hashes.add(pan_hash)

            dup_pan = await documents_coll.find_one({
                "userId": {"$ne": str(current_user["_id"])},
                "panHash": pan_hash
            })

            if dup_pan:
                result["duplicates"].append({
                    "type": "PAN",
                    "value": pan,
                    "conflictWithUser": dup_pan.get("userId")
                })
                result["riskScore"] += 40

            await documents_coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"panHash": pan_hash}}
            )

    # ---- Determine risk level ----
    if result["riskScore"] >= 70:
        result["riskLevel"] = "High"
    elif result["riskScore"] >= 30:
        result["riskLevel"] = "Medium"

    # ---- async update_many ----
    await documents_coll.update_many(
        {"userId": str(current_user["_id"])},
        {"$set": {
            "fraudCheck": result,
            "fraudCheckedAt": datetime.utcnow().isoformat()
        }}
    )

    return JSONResponse(content={"message": "Fraud check complete", "data": result})

# ---------- Milestone2: fraud-score helpers & endpoint ----------

from fuzzywuzzy import fuzz   # or: from rapidfuzz import fuzz as fuzz
import numpy as np
from PIL import Image, ImageFilter

def name_similarity_score(user_name: str, extracted_name: str) -> int:
    """
    Returns fuzzy similarity 0-100. Uses token_sort_ratio style comparison.
    """
    if not user_name or not extracted_name:
        return 0
    try:
        return fuzz.token_sort_ratio(user_name, extracted_name)
    except Exception:
        # fallback to simple ratio
        return fuzz.ratio(user_name, extracted_name)

def is_image_blurry_bytes(image_bytes: bytes, threshold: float = 100.0) -> dict:
    """
    Check blur using variance of Laplacian-like measure via numpy gradient.
    Returns dict {"blurry": bool, "score": float}
    `threshold` can be tuned; lower = more sensitive.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(img, dtype=np.float32)
        # approximate Laplacian variance with second derivative estimate using gradients
        gy, gx = np.gradient(arr)
        g2 = np.gradient(gx)[0] + np.gradient(gy)[1]  # crude second derivative sum
        var = float(np.var(g2))
        blurry = var < threshold
        return {"blurry": blurry, "score": var}
    except Exception as e:
        return {"blurry": False, "score": 0.0}

def calculate_fraud_score_from_signals(verification_valid: bool, duplicate_found: bool, name_sim: int, blurry: bool):
    """
    Returns dict { score: int, riskLevel: str, breakdown: {...} }
    We convert signal booleans/scores to additive points.
    """
    points = 0
    breakdown = {}

    # verification (40 points)
    if not verification_valid:
        points += 40
        breakdown["verification"] = 40
    else:
        breakdown["verification"] = 0

    # duplicate (30 points)
    if duplicate_found:
        points += 30
        breakdown["duplicate"] = 30
    else:
        breakdown["duplicate"] = 0

    # name matching (20 points)
    if name_sim < 70:
        points += 20
        breakdown["name_mismatch"] = 20
    elif name_sim < 85:
        points += 10
        breakdown["name_mismatch"] = 10
    else:
        breakdown["name_mismatch"] = 0

    # image quality (20 points)
    if blurry:
        points += 20
        breakdown["image_quality"] = 20
    else:
        breakdown["image_quality"] = 0

    score = min(points, 100)

    if score <= 30:
        level = "Low"
    elif score <= 70:
        level = "Medium"
    else:
        level = "High"

    return {"score": score, "riskLevel": level, "breakdown": breakdown}

@app.get("/fraud-score", tags=["Fraud Detection / Verification"])
async def fraud_score(current_user: dict = Depends(get_current_user)):
    """
    Compute fraud score for the most recent document of the current user.
    Combines verification result, duplicate detection, name similarity, and image quality.
    """

    # ---- 1. Fetch latest document (ASYNC) ----
    doc = await documents_coll.find_one(
        {"userId": str(current_user["_id"])},
        sort=[("uploadedAt", -1)]
    )

    if not doc:
        raise HTTPException(status_code=404, detail="No documents for this user")

    parsed = doc.get("parsed", {})
    verification = doc.get("verificationResult", {})

    # ---- 2. Verification validity ----
    aadhaar = parsed.get("aadhaarNumber")
    pan = parsed.get("panNumber")

    if verification:
        verification_valid = verification.get("overall", {}).get("valid", False)
    else:
        if aadhaar:
            verification_valid = verify_aadhaar_format(aadhaar)["valid"]
        elif pan:
            verification_valid = verify_pan_format(pan)["valid"]
        else:
            verification_valid = False

    # ---- 3. Duplicate detection (ASYNC) ----
    duplicate_found = False

    if aadhaar:
        ahash = hashlib.sha256(aadhaar.encode()).hexdigest()
        dup = await documents_coll.find_one({
            "userId": {"$ne": str(current_user["_id"])},
            "aadhaarHash": ahash
        })
        duplicate_found = bool(dup)

    elif pan:
        phash = hashlib.sha256(pan.encode()).hexdigest()
        dup = await documents_coll.find_one({
            "userId": {"$ne": str(current_user["_id"])},
            "panHash": phash
        })
        duplicate_found = bool(dup)

    # ---- 4. Name similarity ----
    extracted_name = parsed.get("name") or ""
    user_name = current_user.get("name") or ""
    name_sim = name_similarity_score(user_name, extracted_name)

    # ---- 5. Image quality detection ----
    blurry = False
    img_score = None
    file_path = doc.get("filePath") or doc.get("storedPath")

    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            b = f.read()
        q = is_image_blurry_bytes(b)
        blurry = q["blurry"]
        img_score = q["score"]
    else:
        raw = doc.get("rawText", "")
        if len(raw.strip()) < 20:
            blurry = True
            img_score = 0.0

    # ---- 6. Calculate final fraud score ----
    result = calculate_fraud_score_from_signals(
        verification_valid,
        duplicate_found,
        name_sim,
        blurry
    )

    fraud_summary = {
        "fraudScore": result["score"],
        "riskLevel": result["riskLevel"],
        "breakdown": result["breakdown"],
        "nameSimilarity": name_sim,
        "imageQualityScore": img_score,
        "checkedAt": datetime.utcnow().isoformat()
    }

    # ---- 7. Save fraud summary (ASYNC) ----
    await documents_coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {"fraudSummary": fraud_summary}}
    )

    return JSONResponse(
        content={
            "message": "Fraud score computed",
            "data": {
                "documentId": str(doc["_id"]),
                **fraud_summary
            }
        }
    )
# --- Simple AML blacklist (example) ---
AML_BLACKLIST = {
    "aadhaar": ["123456789012", "266677554433"],  # example blacklisted aadhaars
    "address": []  # you can add blacklisted addresses here
}
async def run_aml_rules(parsed: dict, current_user: dict) -> list:
    """
    Apply simple AML rules and return list of alerts (strings).
    """
    alerts = []

    aadhaar = parsed.get("aadhaarNumber")
    address = parsed.get("address")

    if aadhaar and aadhaar in AML_BLACKLIST.get("aadhaar", []):
        alerts.append("Aadhaar on blacklist")

    # shared-address rule (ASYNC)
    if address:
        addr_count = await documents_coll.count_documents({"parsed.address": address})
        if addr_count >= 3:
            alerts.append("Shared address used by multiple accounts (AML red flag)")

    return alerts

def build_final_decision(fraud_score: int, aml_alerts: list):
    if fraud_score >= 71:
        return "Flagged_for_review", "High"

    if aml_alerts and fraud_score >= 31:
        return "Flagged_for_review", "High"

    if fraud_score >= 31:
        return "Manual_review_recommended", "Medium"

    if aml_alerts:
        return "Manual_review_recommended", "Medium"

    return "Pass", "Low"

@app.post("/kyc/verify", tags=["Compliance / KYC Pipeline"])
async def kyc_verify(doc_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    """
    End-to-end verification pipeline for a given document (or latest for the user).
    """

    # ---------- 1) LOCATE DOCUMENT ----------
    query = {"userId": str(current_user["_id"])}

    if doc_id:
        try:
            query["_id"] = ObjectId(doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        doc = await documents_coll.find_one(query)
    else:
        doc = await documents_coll.find_one(
            query,
            sort=[("uploadedAt", -1)]
        )

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    parsed = doc.get("parsed", {})

    # ---------- 2) VERIFICATION ----------
    aadhaar = parsed.get("aadhaarNumber")
    pan = parsed.get("panNumber")

    aadhaar_ver = verify_aadhaar_format(aadhaar) if aadhaar else {"valid": False, "message": "No Aadhaar found"}
    pan_ver = verify_pan_format(pan) if pan else {"valid": False, "message": "No PAN found"}

    verification_overall_valid = bool(aadhaar_ver.get("valid") or pan_ver.get("valid"))

    # ---------- 3) DUPLICATE DETECTION (ASYNC) ----------
    duplicate_found = False

    if aadhaar:
        ahash = hash_identifier(aadhaar)
        dup = await documents_coll.find_one({
            "userId": {"$ne": str(current_user["_id"])},
            "aadhaarHash": ahash
        })
        duplicate_found = bool(dup)

    elif pan:
        phash = hash_identifier(pan)
        dup = await documents_coll.find_one({
            "userId": {"$ne": str(current_user["_id"])},
            "panHash": phash
        })
        duplicate_found = bool(dup)

    # ---------- 4) NAME SIMILARITY ----------
    extracted_name = parsed.get("name") or ""
    user_name = current_user.get("name") or ""
    name_sim = name_similarity_score(user_name, extracted_name)

    # ---------- 5) IMAGE QUALITY ----------
    file_path = doc.get("filePath")
    image_quality_score = None
    blurry = False

    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            b = f.read()
        q = is_image_blurry_bytes(b)
        image_quality_score = q["score"]
        blurry = q["blurry"]
    else:
        raw = doc.get("rawText", "")
        if len(raw.strip()) < 20:
            blurry = True
            image_quality_score = 0.0

    # ---------- 6) FRAUD SCORE ----------
    fraud_signals = calculate_fraud_score_from_signals(
        verification_overall_valid,
        duplicate_found,
        name_sim,
        blurry
    )

    fraud_score = fraud_signals["score"]
    fraud_risk = fraud_signals["riskLevel"]

    # ---------- 7) AML CHECKS (ASYNC) ----------
    aml_alerts = await run_aml_rules(parsed, current_user)

    # ---------- 8) FINAL DECISION ----------
    finalDecision, finalRisk = build_final_decision(fraud_score, aml_alerts)

    confidence = min(
        0.99,
        max(0.01, fraud_score / 100.0 + (0.1 if aml_alerts else 0.0))
    )

    # ---------- 9) BUILD VERIFICATION LOG ----------
    verification_log = {
        "userId": str(current_user["_id"]),
        "userEmail": current_user.get("email"),
        "documentId": str(doc.get("_id")),
        "filename": doc.get("filename"),
        "timestamp": datetime.utcnow().isoformat(),
        "verificationChecks": {
            "aadhaar": aadhaar_ver,
            "pan": pan_ver,
            "nameSimilarity": name_sim,
            "imageQualityScore": image_quality_score,
            "duplicateFound": duplicate_found
        },
        "fraudScore": fraud_score,
        "fraudRisk": fraud_risk,
        "amlAlerts": aml_alerts,
        "finalDecision": finalDecision,
        "confidence": confidence
    }

    # ---------- 10) SAVE VERIFICATION LOG (ASYNC) ----------
    await verification_logs_coll.insert_one(verification_log)

    # ---------- 11) IF HIGH RISK, CREATE FRAUD ALERT ----------
    if finalRisk == "High" or aml_alerts:
        alert = {
            "documentId": str(doc.get("_id")),
            "userId": str(current_user["_id"]),
            "userEmail": current_user.get("email"),
            "riskLevel": finalRisk,
            "reasons": (
                aml_alerts +
                (["Duplicate Aadhaar/PAN detected"] if duplicate_found else []) +
                (["Name mismatch"] if name_sim < 70 else [])
            ),
            "timestamp": datetime.utcnow().isoformat(),
            "handled": False
        }
        await fraud_alerts_coll.insert_one(alert)

    # ---------- 12) UPDATE DOCUMENT FIELDS (ASYNC) ----------
    await documents_coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "verificationResult": {
                "aadhaar": aadhaar_ver,
                "pan": pan_ver,
                "overall": {"valid": verification_overall_valid}
            },
            "fraudSummary": {
                "fraudScore": fraud_score,
                "fraudRisk": fraud_risk,
                "nameSimilarity": name_sim,
                "imageQualityScore": image_quality_score,
                "checkedAt": datetime.utcnow().isoformat()
            }
        }}
    )

    # ---------- 13) SEND RESPONSE ----------
    return JSONResponse(content={
        "message": "KYC verification pipeline complete",
        "data": {
            "documentId": str(doc["_id"]),
            "identityVerified": verification_overall_valid,
            "fraudScore": fraud_score,
            "fraudRisk": fraud_risk,
            "amlAlerts": aml_alerts,
            "finalDecision": finalDecision,
            "confidence": confidence,
            "checkedAt": datetime.utcnow().isoformat()
        }
    })

# -------------------- UPLOAD DOC --------------------
@app.post("/upload/", tags=["KYC Operations"])
async def upload_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    try:
        start_time = time.time()

        # ---------- 1. SAVE FILE LOCALLY ----------
        save_dir = "./uploads"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, file.filename)

        # async read
        file_bytes = await file.read()

        # write (sync is fine because short operation)
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # ---------- 2. OCR EXTRACTION ----------
        image = Image.open(save_path)
        text = pytesseract.image_to_string(image)
        parsed = parse_text(text)

        # Detect document type
        if parsed.get("aadhaarNumber"):
            doc_type = "Aadhaar"
        elif parsed.get("panNumber"):
            doc_type = "PAN"
        else:
            doc_type = "UNKNOWN"

        # ---------- 3. BUILD RECORD ----------
        record = {
            "userId": str(user["_id"]),
            "filename": file.filename,
            "filePath": save_path,
            "docType": doc_type,
            "rawText": text,
            "parsed": parsed,
            "processingTimeMs": int((time.time() - start_time) * 1000),
            "uploadedAt": datetime.utcnow().isoformat(),
        }

        # ---------- 4. INSERT INTO DOCUMENTS (ASYNC) ----------
        insert_result = await documents_coll.insert_one(record)
        record["_id"] = str(insert_result.inserted_id)

        # ---------- 5. INSERT INTO KYC COLLECTION (ASYNC) ----------
        await kyc_coll.insert_one({
            "userId": str(user["_id"]),
            "docType": doc_type,
            "parsedData": parsed,
            "createdAt": datetime.utcnow().isoformat(),
        })

        # ---------- 6. RETURN ----------
        return JSONResponse(content={"message": "File uploaded successfully", "data": record})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- FETCH DOCS --------------------
@app.get("/api/get-user-docs", tags=["KYC Operations"])
async def get_user_docs(current_user: dict = Depends(get_current_user)):
    cursor = documents_coll.find({"userId": str(current_user["_id"])})

    docs = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this user")

    return {
        "user": current_user["email"],
        "documents": docs
    }
@app.get("/kyc/status", tags=["Compliance / KYC Pipeline"])
async def kyc_status(current_user: dict = Depends(get_current_user)):
    """
    Returns the latest KYC verification status for the current user.
    """
    # Get latest document
    doc = await documents_coll.find_one(
        {"userId": str(current_user["_id"])},
        sort=[("uploadedAt", -1)]
    )

    if not doc:
        raise HTTPException(status_code=404, detail="No documents found for this user")

    # Convert ObjectID → string
    doc["_id"] = str(doc["_id"])

    # Extract stored verification fields
    verification = doc.get("verificationResult", {})
    fraud_summary = doc.get("fraudSummary", {})

    # Get latest verification log
    log = await verification_logs_coll.find_one(
        {"documentId": doc["_id"]},
        sort=[("timestamp", -1)]
    )

    aml_alerts = log.get("amlAlerts") if log else []

    # Build response
    status = {
        "documentId": doc["_id"],
        "filename": doc.get("filename"),
        "uploadedAt": doc.get("uploadedAt"),

        "identityVerified": verification.get("overall", {}).get("valid", False),
        "verificationDetails": verification,

        "fraudScore": fraud_summary.get("fraudScore"),
        "fraudRisk": fraud_summary.get("fraudRisk"),
        "nameSimilarity": fraud_summary.get("nameSimilarity"),
        "imageQualityScore": fraud_summary.get("imageQualityScore"),

        "amlAlerts": aml_alerts or [],
        "finalDecision": log.get("finalDecision") if log else "Unknown",
        "confidence": log.get("confidence") if log else None,
        "lastCheckedAt": fraud_summary.get("checkedAt"),
    }

    return {"message": "KYC status fetched", "data": status}

@app.get("/admin/alerts", tags=["Compliance / Admin"])
async def get_fraud_alerts(current_user: dict = Depends(get_current_user)):
    alerts_cursor = fraud_alerts_coll.find().sort("timestamp", -1)
    alerts = []
    async for a in alerts_cursor:
        a["_id"] = str(a["_id"])
        alerts.append(a)
    return {"alerts": alerts}

@app.get("/admin/logs", tags=["Compliance / Admin"])
async def get_verification_logs(current_user: dict = Depends(get_current_user)):
    logs_cursor = verification_logs_coll.find().sort("timestamp", -1)
    logs = []
    async for log in logs_cursor:
        log["_id"] = str(log["_id"])
        logs.append(log)
    return {"logs": logs}

@app.post("/admin/resolve-alert/{alert_id}", tags=["Compliance / Admin"])
async def resolve_alert(alert_id: str, current_user: dict = Depends(get_current_user)):
    from bson import ObjectId

    try:
        _id = ObjectId(alert_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid alert id")

    result = await fraud_alerts_coll.update_one(
        {"_id": _id},
        {"$set": {"handled": True}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert marked as resolved"}

@app.get("/", tags=["Root"])
def home():
    return {"message": "✅ KYC OCR API (Signup + Login + Upload + OCR) running successfully!"}