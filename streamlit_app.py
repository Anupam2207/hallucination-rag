import streamlit as st

from src.pipeline import HallucinationRAGPipeline
from src.ui.components import render_claims, render_evidence


st.set_page_config(page_title="Hallucination Detection with RAG", layout="wide")


@st.cache_resource
def get_pipeline() -> HallucinationRAGPipeline:
    return HallucinationRAGPipeline()


def render_status(metrics: dict) -> None:
    status = metrics.get("correction_status", "no_change")
    if status == "improved":
        st.success("Correction improved factual grounding or reduced hallucination.")
    elif status == "partially_improved":
        st.warning("Correction partially improved the answer, but one metric may still need review.")
    elif status == "no_change":
        st.warning("Correction preserved the metric score but did not improve it for this query.")
    else:
        st.error("Correction worsened the current metric score for this query.")


def render_metric_explanation() -> None:
    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
- **Raw support**: fraction of raw-answer claims strongly supported by retrieved evidence.
- **Corrected support**: fraction of corrected-answer claims strongly supported by retrieved evidence.
- **Raw weighted / Corrected weighted**: weighted support where supported = 1.0, weak support = 0.5, unsupported = 0.0.
- **Hallucination Δ**: raw hallucination rate minus corrected hallucination rate. Higher is better.
- **Factual improvement**: corrected weighted support minus raw weighted support. Higher is better.
- **Correction status**: improved, partially improved, no change, or worsened based on the two key before/after metrics.
"""
        )


def main() -> None:
    st.title("Hallucination Detection and Correction in LLMs using RAG")
    st.caption(
        "Lightweight M.Tech project demo using local Ollama, ChromaDB, "
        "and claim-level evidence support scoring."
    )

    query = st.text_input("Enter your query", value="What is retrieval-augmented generation?")
    top_k = st.slider("Retrieved passages", min_value=1, max_value=8, value=4)
    run_clicked = st.button("Run pipeline")

    if not run_clicked:
        st.info("Run ingestion and vector indexing first, then click Run pipeline.")
        return

    pipeline = get_pipeline()

    try:
        result = pipeline.run(query, top_k=top_k)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    for warning in result.get("warnings", []):
        st.warning(warning)

    st.caption(f"Retrieval mode: {result.get('retrieval_mode', 'unknown')}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw answer")
        st.write(result["raw_answer"])
    with col2:
        st.subheader("Corrected answer")
        st.write(result["corrected_answer"])

    metrics = result["metrics"]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Raw support", f"{metrics['raw_support_ratio']:.2f}")
    metric_cols[1].metric("Corrected support", f"{metrics['corrected_support_ratio']:.2f}")
    metric_cols[2].metric("Raw weighted", f"{metrics['raw_weighted_support_ratio']:.2f}")
    metric_cols[3].metric("Corrected weighted", f"{metrics['corrected_weighted_support_ratio']:.2f}")
    metric_cols[4].metric("Hallucination Δ", f"{metrics['hallucination_reduction']:.2f}")
    metric_cols[5].metric("Factual improvement", f"{metrics['factual_improvement']:.2f}")

    render_status(metrics)
    render_metric_explanation()

    st.subheader("Retrieved evidence")
    render_evidence(result["evidence"])

    render_claims(result["raw_detection"], "Raw answer claim analysis")
    render_claims(result["corrected_detection"], "Corrected answer claim analysis")


if __name__ == "__main__":
    main()
