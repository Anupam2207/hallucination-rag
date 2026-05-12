import streamlit as st

from src.ui.display_utils import claims_to_dataframe, evidence_to_dataframe


def render_evidence(evidence_list: list[dict]) -> None:
    if not evidence_list:
        st.warning("No evidence was retrieved from the vector database.")
        return
    st.dataframe(evidence_to_dataframe(evidence_list), use_container_width=True)


def render_claims(detection_result: dict, title: str) -> None:
    st.subheader(title)
    if not detection_result.get("claims"):
        st.info("No claim-like sentences were extracted from this answer.")
        return
    st.dataframe(claims_to_dataframe(detection_result["claims"]), use_container_width=True)
