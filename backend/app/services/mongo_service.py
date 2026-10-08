"""MongoDB Atlas Cloud Persistence Service for KisanSeva.

Stores multi-modal diagnostic runs, plant foliage disease diagnoses, crop recommendations,
and Cloudinary CDN asset URLs in MongoDB Atlas collections.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("MongoService")

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://tusharmishraop1_db_user:ECCJYHOUbSBbh9jW@setup.tdbbq7v.mongodb.net/?appName=Setup",
)
DB_NAME = os.getenv("MONGODB_DB_NAME", "kisanseva")

_MONGO_CLIENT = None
_MONGO_DB = None
_CONNECTION_CHECKED = False
_IS_CONNECTED = False


def get_mongo_db():
    """Retrieve singleton MongoDB database instance with safe error handling."""
    global _MONGO_CLIENT, _MONGO_DB, _CONNECTION_CHECKED, _IS_CONNECTED

    if _MONGO_DB is not None and _IS_CONNECTED:
        return _MONGO_DB

    if _CONNECTION_CHECKED and not _IS_CONNECTED:
        return None

    try:
        import pymongo
        import certifi

        _MONGO_CLIENT = pymongo.MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            tlsCAFile=certifi.where(),
        )
        # Test connection ping
        _MONGO_CLIENT.admin.command("ping")
        _MONGO_DB = _MONGO_CLIENT[DB_NAME]
        _IS_CONNECTED = True
        logger.info(f"Connected to MongoDB Atlas cluster (Database: '{DB_NAME}').")
        return _MONGO_DB
    except Exception as exc:
        _IS_CONNECTED = False
        _CONNECTION_CHECKED = True
        logger.warning(
            f"MongoDB Atlas is currently unavailable ({exc}). "
            "Falling back to local SQLite storage. (Tip: Ensure IP 0.0.0.0/0 is allowed in Atlas Network Access)."
        )
        return None


def is_mongo_connected() -> bool:
    """Check if MongoDB Atlas is currently connected."""
    db = get_mongo_db()
    return db is not None


def save_plant_diagnosis_to_mongo(record: Dict[str, Any]) -> bool:
    """Save plant disease diagnosis to MongoDB Atlas 'plant_diagnoses' collection."""
    db = get_mongo_db()
    if db is None:
        return False

    try:
        doc = dict(record)
        doc["created_at"] = doc.get("predicted_at") or datetime.now(timezone.utc).isoformat()
        db["plant_diagnoses"].insert_one(doc)
        return True
    except Exception as exc:
        logger.warning(f"Failed to write plant diagnosis to MongoDB: {exc}")
        return False


def save_crop_recommendation_to_mongo(record: Dict[str, Any]) -> bool:
    """Save crop recommendation to MongoDB Atlas 'crop_recommendations' collection."""
    db = get_mongo_db()
    if db is None:
        return False

    try:
        doc = dict(record)
        doc["created_at"] = doc.get("recommended_at") or datetime.now(timezone.utc).isoformat()
        db["crop_recommendations"].insert_one(doc)
        return True
    except Exception as exc:
        logger.warning(f"Failed to write crop recommendation to MongoDB: {exc}")
        return False


def save_analysis_record_to_mongo(record: Dict[str, Any]) -> bool:
    """Save multi-modal analysis record to MongoDB Atlas 'analysis_records' collection."""
    db = get_mongo_db()
    if db is None:
        return False

    try:
        doc = dict(record)
        doc["created_at"] = doc.get("timestamp") or datetime.now(timezone.utc).isoformat()
        db["analysis_records"].insert_one(doc)
        return True
    except Exception as exc:
        logger.warning(f"Failed to write analysis record to MongoDB: {exc}")
        return False


def get_recent_mongo_scans(collection_name: str = "analysis_records", limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent scan documents from MongoDB Atlas."""
    db = get_mongo_db()
    if db is None:
        return []

    try:
        cursor = db[collection_name].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return list(cursor)
    except Exception as exc:
        logger.warning(f"Failed to read from MongoDB '{collection_name}': {exc}")
        return []
