"""
FixMate: AI Repair Assistant
A beginner-friendly Streamlit web application powered by Google Gemini
for diagnosing common household and personal electronic device issues safely.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
import streamlit as st


# ---------------------------------------------------------------------------
# Page Configuration (Must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FixMate: AI Repair Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Constants & System Prompt
# ---------------------------------------------------------------------------
DEVICE_OPTIONS: List[str] = [
    "Phone charger",
    "Fan",
    "Laptop",
    "Mixer",
    "TV remote",
    "Other",
]

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


# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Injects clean, theme-aware custom CSS for cards, badges, and warnings."""
    st.markdown(
        """
        <style>
        /* Main header styling */
        .fixmate-header {
            padding: 1.1rem 1.4rem;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(16, 185, 129, 0.10));
            border: 1px solid rgba(128, 128, 128, 0.22);
            margin-bottom: 1.25rem;
        }
        .fixmate-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
        }
        .fixmate-header p {
            margin: 0.35rem 0 0 0;
            opacity: 0.85;
            font-size: 0.98rem;
        }

        /* Diagnostic section cards */
        .diag-card {
            padding: 1rem 1.15rem;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background-color: rgba(128, 128, 128, 0.04);
            margin-bottom: 0.85rem;
        }
        .diag-section-title {
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* Ranked cause items */
        .cause-item {
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            background-color: rgba(59, 130, 246, 0.06);
            margin-bottom: 0.5rem;
        }
        .cause-rank-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            background-color: #2563eb;
            color: #ffffff;
            margin-right: 0.45rem;
        }

        /* Tool pills */
        .tool-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 500;
            border: 1px solid rgba(128, 128, 128, 0.3);
            background-color: rgba(128, 128, 128, 0.08);
        }

        /* Safety warning callout */
        .safety-box {
            padding: 0.9rem 1.1rem;
            border-radius: 10px;
            border-left: 5px solid #dc2626;
            background-color: rgba(220, 38, 38, 0.10);
            margin-bottom: 0.85rem;
        }
        .safety-box-title {
            font-weight: 700;
            color: #ef4444;
            margin-bottom: 0.3rem;
            font-size: 0.98rem;
        }

        /* Clarifying questions callout */
        .question-box {
            padding: 0.85rem 1.1rem;
            border-radius: 10px;
            border-left: 5px solid #8b5cf6;
            background-color: rgba(139, 92, 246, 0.09);
            margin-bottom: 0.85rem;
        }

        /* Verdict badge container */
        .verdict-container {
            padding: 1rem 1.2rem;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: rgba(128, 128, 128, 0.05);
            margin-top: 0.4rem;
        }
        .verdict-badge {
            display: inline-block;
            padding: 0.38rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
            margin-right: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .badge-safe {
            background-color: #16a34a;
            color: #ffffff;
        }
        .badge-caution {
            background-color: #d97706;
            color: #ffffff;
        }
        .badge-tech {
            background-color: #dc2626;
            color: #ffffff;
        }
        .badge-sub {
            display: inline-block;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.84rem;
            border: 1px solid rgba(128, 128, 128, 0.35);
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API & Session State Helpers
# ---------------------------------------------------------------------------
def get_api_key() -> Optional[str]:
    """Reads the Gemini API key from environment variables."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def initialize_session_state() -> None:
    """Initializes required keys in Streamlit session state."""
    if "messages" not in st.session_state:
        # Each item: {"role": "user"|"assistant", "content": str|dict, "image": Optional[Image.Image]}
        st.session_state.messages = []
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "active_device" not in st.session_state:
        st.session_state.active_device = None


def reset_conversation() -> None:
    """Clears the current diagnosis, chat history, and Gemini chat session."""
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.active_device = None


def create_gemini_chat(api_key: str) -> Any:
    """Configures the Gemini SDK and initializes a new multimodal chat session."""
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
    return model.start_chat(history=[])


def parse_fixmate_response(raw_text: str) -> Dict[str, Any]:
    """
    Safely parses the structured JSON response from Gemini.
    Includes fallback extraction in case markdown fences wrap the JSON.
    """
    cleaned = raw_text.strip()

    # Strip optional markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        # Attempt to extract the first JSON object found in the string
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    # Graceful fallback structure if raw text is non-JSON
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


def load_uploaded_image(uploaded_file: Any) -> Optional[Image.Image]:
    """Safely opens an uploaded image file as a PIL Image."""
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file)
        image.load()  # Verify image integrity
        return image
    except (UnidentifiedImageError, OSError, Exception) as exc:
        st.error(
            f"Could not read the uploaded image ({exc}). "
            "Please upload a valid PNG, JPG, or WEBP photo."
        )
        return None


# ---------------------------------------------------------------------------
# UI Rendering Functions
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Renders the sidebar with the mandatory safety disclaimer and controls."""
    with st.sidebar:
        st.markdown("## 🛡️ Safety Disclaimer")
        st.warning(
            "**Read Before Starting Any Check:**\n\n"
            "- **Unplug First:** Always disconnect the device from wall power and remove batteries (if safe) before inspecting.\n"
            "- **High-Risk Components:** **NEVER** open power supplies/adapters, microwave ovens, or handle swollen/leaking batteries.\n"
            "- **Stop Immediately:** If you smell burning, see smoke/sparks, or notice frayed mains cords, stop and call a licensed technician.\n"
            "- **Educational Guidance:** FixMate provides beginner troubleshooting tips and is not a substitute for professional electrical repair."
        )

        st.markdown("---")
        st.markdown("### 📸 Photo Tips for Better Results")
        st.caption(
            "• Use bright, clear lighting.\n"
            "• Capture damaged cables, ports, or indicator lights.\n"
            "• Include model/rating labels if visible."
        )

        st.markdown("---")
        if st.button("🔄 Start New Diagnosis", use_container_width=True):
            reset_conversation()
            st.rerun()


def render_verdict_badge(
    verdict: str, category: str, reason: str
) -> None:
    """Renders the colored Verdict badge and one-line explanation."""
    verdict_clean = (verdict or "Try with caution").strip()
    verdict_lower = verdict_clean.lower()

    if "safe" in verdict_lower:
        badge_class = "badge-safe"
        icon = "✅"
        default_cat = "Fix at home"
    elif "technician" in verdict_lower or "call" in verdict_lower:
        badge_class = "badge-tech"
        icon = "🚨"
        default_cat = "Call a technician"
    else:
        badge_class = "badge-caution"
        icon = "⚠️"
        default_cat = "Fix at home"

    category_clean = (category or default_cat).strip()
    reason_clean = (reason or "Follow the safety steps above before proceeding.").strip()

    st.markdown(
        f"""
        <div class="verdict-container">
            <div class="diag-section-title">🏁 Verdict</div>
            <div>
                <span class="verdict-badge {badge_class}">{icon} {verdict_clean}</span>
                <span class="badge-sub">📌 {category_clean}</span>
            </div>
            <p style="margin: 0.45rem 0 0 0; font-size: 0.96rem;"><strong>Why:</strong> {reason_clean}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnostic_response(data: Dict[str, Any]) -> None:
    """Renders the structured FixMate diagnostic report in clean visual sections."""
    # 1. Safety Warning (Always prominently visible)
    safety_warning = data.get(
        "safety_warning",
        "Always unplug the device first. Never open power supplies, microwaves, or swollen batteries.",
    )
    st.markdown(
        f"""
        <div class="safety-box">
            <div class="safety-box-title">⚠️ Safety Warning</div>
            <div style="font-size: 0.94rem;">{safety_warning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Likely Causes (Ranked)
    causes = data.get("likely_causes", [])
    if causes:
        st.markdown('<div class="diag-section-title">🔎 Likely Causes (Ranked)</div>', unsafe_allow_html=True)
        for idx, item in enumerate(causes, start=1):
            if isinstance(item, dict):
                rank = item.get("rank", idx)
                cause_title = item.get("cause", f"Possible Cause #{rank}")
                explanation = item.get("explanation", "")
            else:
                rank = idx
                cause_title = str(item)
                explanation = ""

            rank_label = "Most Probable" if rank == 1 else f"#{rank}"
            st.markdown(
                f"""
                <div class="cause-item">
                    <div>
                        <span class="cause-rank-badge">{rank_label}</span>
                        <strong>{cause_title}</strong>
                    </div>
                    <div style="margin-top: 0.25rem; font-size: 0.92rem; opacity: 0.9;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Tools Needed
    tools = data.get("tools_needed", [])
    if tools:
        st.markdown('<div class="diag-section-title" style="margin-top: 0.85rem;">🧰 Tools Needed</div>', unsafe_allow_html=True)
        pills_html = "".join(f'<span class="tool-pill">🔧 {tool}</span>' for tool in tools)
        st.markdown(f'<div style="margin-bottom: 0.85rem;">{pills_html}</div>', unsafe_allow_html=True)

    # 4. Step-by-Step Checks
    steps = data.get("step_by_step_checks", [])
    if steps:
        st.markdown('<div class="diag-section-title">📋 Step-by-Step Checks</div>', unsafe_allow_html=True)
        with st.container(border=True):
            for step_num, step_text in enumerate(steps, start=1):
                st.markdown(f"**{step_num}.** {step_text}")

    # 5. Clarifying Questions (if FixMate needs extra info)
    questions = [q for q in data.get("clarifying_questions", []) if str(q).strip()]
    if questions:
        q_items = "".join(f"<li style='margin-bottom: 0.25rem;'>{q}</li>" for q in questions[:2])
        st.markdown(
            f"""
            <div class="question-box">
                <div class="diag-section-title">💬 Quick Follow-Up Questions</div>
                <ul style="margin: 0.2rem 0 0.2rem 1.2rem; padding: 0; font-size: 0.94rem;">
                    {q_items}
                </ul>
                <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.35rem;">
                    Reply in the chat box below to narrow down the diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Verdict with Colored Badge
    render_verdict_badge(
        verdict=data.get("verdict", "Try with caution"),
        category=data.get("verdict_category", ""),
        reason=data.get("verdict_reason", ""),
    )


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()
    initialize_session_state()
    render_sidebar()

    # Header Banner
    st.markdown(
        """
        <div class="fixmate-header">
            <h1>🛠️ FixMate: AI Repair Assistant</h1>
            <p>Beginner-friendly device diagnostics, safe step-by-step checks, and honest repair verdicts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check API key availability early so user gets clear feedback
    api_key = get_api_key()
    if not api_key:
        st.error(
            "**Missing API Key:** Could not find `GEMINI_API_KEY` in your environment variables.\n\n"
            "Set `GEMINI_API_KEY` in your terminal before launching Streamlit (see `README.md` for commands)."
        )

    # -----------------------------------------------------------------------
    # Device Diagnostic Input Panel
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("1. Tell FixMate About Your Device")
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            selected_device = st.selectbox(
                "Select device type",
                options=DEVICE_OPTIONS,
                index=0,
                help="Choose the category that best matches your faulty device.",
            )

            custom_device = ""
            if selected_device == "Other":
                custom_device = st.text_input(
                    "Specify device name (optional)",
                    placeholder="e.g., Hair dryer, Table lamp, Bluetooth speaker",
                )

            problem_description = st.text_area(
                "Describe the problem",
                height=120,
                placeholder=(
                    "Explain what is happening, any unusual sounds/smells, or when it stopped working...\n"
                    "Example: 'My phone charger only works if I hold the wire at an angle and feels warm.'"
                ),
            )

        with col_right:
            uploaded_file = st.file_uploader(
                "Upload a photo of the faulty device (optional)",
                type=["png", "jpg", "jpeg", "webp"],
                help="A clear photo helps FixMate spot frayed wires, bent pins, or corrosion.",
            )
            preview_image: Optional[Image.Image] = None
            if uploaded_file is not None:
                preview_image = load_uploaded_image(uploaded_file)
                if preview_image is not None:
                    st.image(
                        preview_image,
                        caption="Uploaded device photo",
                        use_container_width=True,
                    )

        diagnose_clicked = st.button(
            "🔍 Diagnose",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Handle Initial "Diagnose" Button Submission
    # -----------------------------------------------------------------------
    if diagnose_clicked:
        if not api_key:
            st.error("Please configure your `GEMINI_API_KEY` environment variable first.")
        elif not problem_description or not problem_description.strip():
            st.warning("⚠️ Please describe the problem in the text box before clicking **Diagnose**.")
        else:
            device_label = (
                f"Other ({custom_device.strip()})"
                if selected_device == "Other" and custom_device.strip()
                else selected_device
            )

            user_prompt = (
                f"Device Type: {device_label}\n"
                f"Problem Description: {problem_description.strip()}\n"
                f"Photo Attached: {'Yes' if preview_image is not None else 'No'}\n\n"
                "Please diagnose this issue following all FixMate safety rules and return the structured JSON response."
            )

            try:
                with st.spinner("🛠️ FixMate is inspecting your device details and checking safety rules..."):
                    chat = create_gemini_chat(api_key)
                    message_parts: List[Any] = [user_prompt]
                    if preview_image is not None:
                        message_parts.append(preview_image)

                    response = chat.send_message(message_parts)
                    if not response or not getattr(response, "text", None):
                        raise ValueError("Received an empty response from the Gemini API.")

                    parsed_data = parse_fixmate_response(response.text)

                    # Store fresh session state for follow-up conversation
                    st.session_state.chat_session = chat
                    st.session_state.active_device = device_label
                    st.session_state.messages = [
                        {
                            "role": "user",
                            "content": f"**Device:** {device_label}\n\n**Issue:** {problem_description.strip()}",
                            "image": preview_image,
                        },
                        {
                            "role": "assistant",
                            "content": parsed_data,
                            "image": None,
                        },
                    ]
            except Exception as exc:
                st.error(
                    f"❌ **Diagnosis Failed:** An error occurred while communicating with Gemini: `{exc}`\n\n"
                    "Please check your API key, internet connection, or try again in a moment."
                )

    # -----------------------------------------------------------------------
    # Render Diagnostic Results & Follow-Up Chat History
    # -----------------------------------------------------------------------
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("2. Diagnostic Report & Follow-Up Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                    if msg.get("image") is not None:
                        st.image(msg["image"], width=260, caption="Attached photo")
                else:
                    if isinstance(msg["content"], dict):
                        render_diagnostic_response(msg["content"])
                    else:
                        st.markdown(str(msg["content"]))

    # -----------------------------------------------------------------------
    # Follow-Up Chat Input (Allows replying to clarifying questions)
    # -----------------------------------------------------------------------
    if st.session_state.chat_session is not None:
        follow_up = st.chat_input(
            "Reply to FixMate's questions or share what happened after checking..."
        )
        if follow_up and follow_up.strip():
            if not api_key:
                st.error("Missing `GEMINI_API_KEY`. Please set it and restart the app.")
            else:
                user_reply = follow_up.strip()
                st.session_state.messages.append(
                    {"role": "user", "content": user_reply, "image": None}
                )
                with st.chat_message("user"):
                    st.markdown(user_reply)

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("FixMate is updating your diagnosis..."):
                            followup_prompt = (
                                f"User follow-up answer / update: {user_reply}\n\n"
                                "Update the diagnosis, clarifying questions (if any remain), safety warning, "
                                "steps, and verdict based on this new information. Return valid JSON."
                            )
                            response = st.session_state.chat_session.send_message(followup_prompt)
                            if not response or not getattr(response, "text", None):
                                raise ValueError("Received an empty response from Gemini.")

                            updated_data = parse_fixmate_response(response.text)
                            render_diagnostic_response(updated_data)
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": updated_data,
                                    "image": None,
                                }
                            )
                    except Exception as exc:
                        st.error(
                            f"❌ **Follow-Up Error:** Could not get an updated response (`{exc}`). "
                            "Please try sending your message again."
                        )


if __name__ == "__main__":
    main()