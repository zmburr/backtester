"""
Persistent chart annotations backed by a JSON file.
Supports text labels pinned to specific dates on the chart,
with add/edit/remove management.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_PATH = REPO_ROOT / "data" / "chart_annotations.json"


def _load_all() -> Dict[str, list]:
    """Load full annotations dict from JSON."""
    if ANNOTATIONS_PATH.exists():
        try:
            with open(ANNOTATIONS_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_all(data: Dict[str, list]):
    """Save full annotations dict to JSON."""
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def load_annotations(ticker: str) -> List[dict]:
    """Load annotations for a specific ticker.

    Returns list of {"id", "date", "text"} dicts.
    """
    return _load_all().get(ticker.upper(), [])


def add_annotation(ticker: str, date_str: str, text: str) -> str:
    """Add a new annotation. Returns the new annotation ID."""
    data = _load_all()
    key = ticker.upper()
    if key not in data:
        data[key] = []
    ann_id = uuid.uuid4().hex[:8]
    data[key].append({"id": ann_id, "date": date_str, "text": text})
    _save_all(data)
    return ann_id


def update_annotation(ticker: str, ann_id: str, new_text: str):
    """Update the text of an existing annotation."""
    data = _load_all()
    key = ticker.upper()
    for ann in data.get(key, []):
        if ann["id"] == ann_id:
            ann["text"] = new_text
            break
    _save_all(data)


def remove_annotation(ticker: str, ann_id: str):
    """Remove an annotation by ID."""
    data = _load_all()
    key = ticker.upper()
    data[key] = [a for a in data.get(key, []) if a["id"] != ann_id]
    if not data[key]:
        del data[key]
    _save_all(data)


# ---------------------------------------------------------------------------
# UI — annotation manager below chart
# ---------------------------------------------------------------------------

def render_annotation_manager(ticker: str, chart_dates: list):
    """
    Render the annotation management panel.

    Shows:
    - Date selector + text input to add new annotations
    - List of existing annotations with edit/remove controls
    """
    annotations = load_annotations(ticker)

    # -- Add new annotation --
    st.markdown(
        '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
        'text-transform:uppercase; letter-spacing:0.1em; color:#6b7a90;">Add Annotation</span>',
        unsafe_allow_html=True,
    )

    # Use chart-clicked date if available
    default_idx = 0
    clicked_date = st.session_state.get("chart_clicked_date")
    if clicked_date and clicked_date in chart_dates:
        default_idx = chart_dates.index(clicked_date)

    col_date, col_text, col_btn = st.columns([2, 5, 1])
    with col_date:
        ann_date = st.selectbox(
            "Date",
            options=chart_dates,
            index=default_idx,
            key="ann_date_select",
            label_visibility="collapsed",
        )
    with col_text:
        ann_text = st.text_input(
            "Annotation text",
            value="",
            key="ann_text_input",
            label_visibility="collapsed",
            placeholder="Type annotation...",
        )
    with col_btn:
        if st.button("Pin", key="ann_add_btn") and ann_text.strip() and ann_date:
            add_annotation(ticker, ann_date, ann_text.strip())
            st.rerun()

    # -- Existing annotations --
    if not annotations:
        return

    st.markdown(
        '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
        'text-transform:uppercase; letter-spacing:0.1em; color:#6b7a90;">'
        f'Annotations ({len(annotations)})</span>',
        unsafe_allow_html=True,
    )

    for ann in sorted(annotations, key=lambda a: a["date"], reverse=True):
        col_info, col_edit, col_del = st.columns([6, 1, 1])
        with col_info:
            st.markdown(
                f'<span style="font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#4fc3f7;">'
                f'{ann["date"]}</span> '
                f'<span style="font-size:0.82rem; color:#c0c8d8;">{ann["text"]}</span>',
                unsafe_allow_html=True,
            )
        with col_edit:
            if st.button("Edit", key=f"ann_edit_{ann['id']}"):
                st.session_state[f"editing_{ann['id']}"] = True
        with col_del:
            if st.button("Del", key=f"ann_del_{ann['id']}"):
                remove_annotation(ticker, ann["id"])
                st.rerun()

        # Inline edit form
        if st.session_state.get(f"editing_{ann['id']}"):
            edit_col1, edit_col2 = st.columns([5, 1])
            with edit_col1:
                new_text = st.text_input(
                    "Edit",
                    value=ann["text"],
                    key=f"ann_edittext_{ann['id']}",
                    label_visibility="collapsed",
                )
            with edit_col2:
                if st.button("Save", key=f"ann_save_{ann['id']}"):
                    update_annotation(ticker, ann["id"], new_text)
                    del st.session_state[f"editing_{ann['id']}"]
                    st.rerun()
