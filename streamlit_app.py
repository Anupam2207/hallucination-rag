import streamlit as st

from src.pipeline import HallucinationRAGPipeline
from src.ui.components import render_claims, render_evidence


st.set_page_config(page_title="Hallucination Detection with RAG", layout="wide")


@st.cache_resource
def get_pipeline() -> HallucinationRAGPipeline:
    return HallucinationRAGPipeline()


def main() -> None:
    st.title("Hallucination Detection and Correction in LLMs using RAG")
    st.caption("Lightweight M.Tech project demo using local Ollama + ChromaDB + similarity-based claim support scoring")

    query = st.text_input("Enter your query", value="What is retrieval-augmented generation?")
    run_clicked = st.button("Run pipeline")

    if not run_clicked:
        st.info("Build the knowledge base and vector index first, then run the pipeline.")
        return

    pipeline = get_pipeline()

    try:
        result = pipeline.run(query)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    for warning in result.get("warnings", []):
        st.warning(warning)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw answer")
        st.write(result["raw_answer"])
    with col2:
        st.subheader("Corrected answer")
        st.write(result["corrected_answer"])

    metrics = result["metrics"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Raw support ratio", f"{metrics['raw_support_ratio']:.2f}")
    metric_cols[1].metric("Corrected support ratio", f"{metrics['corrected_support_ratio']:.2f}")
    metric_cols[2].metric("Raw hallucination rate", f"{metrics['raw_hallucination_rate']:.2f}")
    metric_cols[3].metric("Corrected hallucination rate", f"{metrics['corrected_hallucination_rate']:.2f}")
    metric_cols[4].metric("Factual improvement", f"{metrics['factual_improvement']:.2f}")

    st.subheader("Retrieved evidence")
    render_evidence(result["evidence"])

    render_claims(result["raw_detection"], "Raw answer claim analysis")
    render_claims(result["corrected_detection"], "Corrected answer claim analysis")


if __name__ == "__main__":
    main()
