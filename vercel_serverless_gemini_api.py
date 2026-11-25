"""
Vercel Python Serverless Function for FixMate: AI Repair Assistant.
Endpoint: POST /api/diagnose
Reads GEMINI_API_KEY from Vercel Environment Variables and returns structured JSON diagnostics.
"""

import base64
from http.server import BaseHTTPRequestHandler
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image, UnidentifiedImageError


MODEL_NAME: str = "gemini-2.5-flash"

SYSTEM_PROMPT: str = """
You are FixMate, a careful repair assistant for beginners.
Given a device type, a problem description, and optionally a photo:
1. Ask up to 2 clarifying questions if details are missing.
2. List 3 likely causes, most probable first.
3. Give simple, numbered troubleshooting steps and the tools needed.
4. Always start with: unplug the device and remove the battery if safe.
5. Never suggest opening power supplies, microwaves, or swollen batteries.
6. End with a verdict: "Safe to try at home", "Try with caution", or "Take to a technician", and explain why in one line.
Use simple language and short steps.

CRITICAL OUTPUT FORMAT:
You MUST respond ONLY with a valid JSON object using this exact schema (both for the initial diagnosis and for all follow-up replies):
{
  "clarifying_questions": [
    "Up to 2 short clarifying questions if details are missing, or an empty array [] if enough info is available."
  ],
  "likely_causes": [
    {
      "rank": 1,
      "cause": "Most probable cause title",
      "explanation": "1-2 short, beginner-friendly sentences explaining why."
    },
    {
      "rank": 2,
      "cause": "Second most probable cause title",
      "explanation": "1-2 short, beginner-friendly sentences explaining why."
    },
    {
      "rank": 3,
      "cause": "Third most probable cause title",
      "explanation": "1-2 short, beginner-friendly sentences explaining why."
    }
  ],
  "tools_needed": [
    "Simple tool or household item (e.g., Dry cloth, Flashlight, or 'No special tools needed')"
  ],
  "step_by_step_checks": [
    "Unplug the device and remove the battery if safe.",
    "Short, simple step 2...",
    "Short, simple step 3..."
  ],
  "safety_warning": "Clear safety warning tailored to this device and symptom. Explicitly remind the user never to open power supplies, microwaves, or swollen batteries.",
  "verdict": "Safe to try at home | Try with caution | Take to a technician",
  "verdict_category": "Fix at home | Call a technician",
  "verdict_reason": "One clear line explaining why this verdict applies."
}

Rules for verdict mapping:
- If verdict is "Safe to try at home" or "Try with caution", set verdict_category to "Fix at home".
- If verdict is "Take to a technician" (or if there is any sign of swollen battery, burning smell, exposed mains wiring, or internal power supply failure), set verdict_category to "Call a technician".
- Always keep step_by_step_checks[0] strictly focused on unplugging the device and removing the battery if safe.
"""


def get_api_key() -> Optional[str]:
    """Reads and sanitizes the Gemini API key from environment variables."""
    raw_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not raw_key:
        return None
    cleaned = raw_key.strip().strip('"').strip("'")
    return cleaned or None


def parse_fixmate_response(raw_text: str) -> Dict[str, Any]:
    """Safely parses the structured JSON response from Gemini."""
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    return {
        "clarifying_questions": [],
        "likely_causes": [],
        "tools_needed": [],
        "step_by_step_checks": [
            "Unplug the device and remove the battery if safe.",
            cleaned,
        ],
        "safety_warning": "Never open sealed power supplies, microwaves, or swollen batteries.",
        "verdict": "Try with caution",
        "verdict_category": "Fix at home",
        "verdict_reason": "Review the guidance carefully and consult a technician if unsure.",
    }


def decode_base64_image(b64_data: str) -> Image.Image:
    """Decodes a base64 image string into a verified PIL Image."""
    raw_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw_bytes))
    img.load()
    return img


def build_gemini_contents(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts client conversation turns into Gemini multi-turn content format."""
    contents: List[Dict[str, Any]] = []
    for turn in turns:
        role = turn.get("role", "user")
        if role not in ("user", "model"):
            role = "user"

        parts: List[Any] = []
        text = str(turn.get("text", "")).strip()
        if text:
            parts.append(text)

        image_b64 = turn.get("image_base64")
        if role == "user" and image_b64:
            pil_image = decode_base64_image(str(image_b64))
            parts.append(pil_image)

        if parts:
            contents.append({"role": role, "parts": parts})

    return contents


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless HTTP Request Handler."""

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        api_key = get_api_key()
        if not api_key:
            self._send_json(
                500,
                {
                    "error": (
                        "Server is missing GEMINI_API_KEY. Add GEMINI_API_KEY in Vercel "
                        "Project Settings -> Environment Variables and redeploy."
                    )
                },
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json(400, {"error": "Empty request body."})
                return

            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Invalid JSON request payload: {exc}"})
            return

        turns = payload.get("turns")
        if not isinstance(turns, list) or not turns:
            self._send_json(400, {"error": "Missing conversation turns in request."})
            return

        try:
            contents = build_gemini_contents(turns)
            if not contents:
                self._send_json(400, {"error": "No valid message content provided."})
                return

            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
                generation_config=generation_config,
            )

            response = model.generate_content(contents)
            if not response or not getattr(response, "text", None):
                raise ValueError("Received an empty response from the Gemini API.")

            parsed_result = parse_fixmate_response(response.text)
            self._send_json(200, {"result": parsed_result})

        except (UnidentifiedImageError, base64.binascii.Error) as exc:
            self._send_json(400, {"error": f"Invalid uploaded image data: {exc}"})
        except Exception as exc:
            self._send_json(500, {"error": f"Gemini diagnosis error: {exc}"})