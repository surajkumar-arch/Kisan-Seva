"""Browser text-to-speech for the Streamlit dashboard.

Uses the Web Speech API (``speechSynthesis``) which runs on the user's device,
needs no API key and no server round-trip. Voice availability for Indian
languages depends on the voices installed in the user's browser/OS.
"""

from __future__ import annotations

import json
import re

import streamlit.components.v1 as components

LANG_CODES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
}


def clean_for_speech(text: str) -> str:
    """Strip markdown so the voice does not read symbols aloud."""
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def read_aloud_button(text: str, language: str = "English") -> None:
    """Render 'Read aloud' / 'Stop' buttons that speak ``text`` in the browser."""
    payload = json.dumps(clean_for_speech(text))
    lang = json.dumps(LANG_CODES.get(language, "en-IN"))
    components.html(
        f"""
        <div style="font-family:system-ui,sans-serif;display:flex;gap:8px;align-items:center">
          <button id="speak" style="padding:6px 14px;border-radius:8px;border:1px solid #16a34a;
                  background:#f0fdf4;color:#14532d;font-weight:600;cursor:pointer">🔊 Read aloud</button>
          <button id="stop" style="padding:6px 12px;border-radius:8px;border:1px solid #d1d5db;
                  background:#fff;color:#374151;cursor:pointer">⏹ Stop</button>
          <span id="msg" style="font-size:12px;color:#6b7280"></span>
        </div>
        <script>
          const text = {payload}, lang = {lang};
          const msg = document.getElementById('msg');
          document.getElementById('speak').onclick = () => {{
            if (!('speechSynthesis' in window)) {{ msg.textContent = 'Speech not supported in this browser'; return; }}
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = lang;
            const v = window.speechSynthesis.getVoices().find(v => v.lang === lang)
                   || window.speechSynthesis.getVoices().find(v => v.lang.startsWith(lang.split('-')[0]));
            if (v) u.voice = v; else msg.textContent = 'No ' + lang + ' voice installed; using default';
            window.speechSynthesis.speak(u);
          }};
          document.getElementById('stop').onclick = () => window.speechSynthesis.cancel();
        </script>
        """,
        height=48,
    )
