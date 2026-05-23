import streamlit as st

from src.ui.display_utils import claims_to_dataframe, evidence_to_dataframe


def render_evidence(evidence_list: list[dict]) -> None:
    if not evidence_list:
        st.warning("No evidence was retrieved from the vector database.")
        return
    st.dataframe(evidence_to_dataframe(evidence_list), use_container_width=True)
    with st.expander("Full retrieved evidence"):
        for index, item in enumerate(evidence_list, start=1):
            evidence_id = item.get("evidence_id") or f"Evidence-{index}"
            source = (item.get("metadata") or {}).get("source_rel") or item.get("chunk_id", "unknown")
            st.markdown(f"**{evidence_id} — {source}**")
            st.write(item.get("text", ""))
            st.divider()


def render_claims(detection_result: dict, title: str) -> None:
    st.subheader(title)
    claims = detection_result.get("claims", [])
    if not claims:
        st.info("No claim-like sentences were extracted from this answer.")
        return

    st.dataframe(claims_to_dataframe(claims, compact=True), use_container_width=True)

    with st.expander(f"Detailed verification view for {title}"):
        for index, item in enumerate(claims, start=1):
            st.markdown(f"**Claim {index}:** {item.get('highlighted_claim') or item.get('claim', '')}")
            st.write(
                {
                    "label": item.get("label"),
                    "support_score": item.get("support_score"),
                    "raw_similarity": item.get("raw_similarity_score"),
                    "rule_flags": item.get("rule_flags", []),
                    "nli_label": item.get("nli_label"),
                    "nli_score": item.get("nli_score"),
                    "nli_available": item.get("nli_available"),
                    "hallucinated_spans": item.get("hallucinated_spans", []),
                }
            )
            evidence_text = item.get("best_evidence_text")
            if evidence_text:
                st.caption("Best supporting evidence")
                st.write(evidence_text)
            st.divider()
