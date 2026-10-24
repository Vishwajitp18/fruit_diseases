import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file from backend directory if present
load_dotenv()


def _read_cors_origins() -> tuple[str, ...]:
    configured_origins = os.getenv("CORS_ORIGINS")
    if not configured_origins:
        return ("http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000")

    origins = tuple(
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    )
    if not origins:
        raise ValueError("CORS_ORIGINS must contain at least one origin.")
    return origins


@dataclass(frozen=True)
class Settings:
    api_title: str = "Fruit Disease Prediction API"
    api_version: str = "0.3.0"
    cors_origins: tuple[str, ...] = _read_cors_origins()
    max_image_mb: int = 10
    
    # Optional Groq Report Generation
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    enable_groq_reports: bool = os.getenv("ENABLE_GROQ_REPORTS", "false").strip().lower() in ("true", "1", "yes")

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * 1024 * 1024


settings = Settings()
