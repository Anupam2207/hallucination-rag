import streamlit as st

from src.generation.base_answer import BaseAnswerGenerator
from src.generation.correction import AnswerCorrector

from src.retrieval.retriever import SemanticRetriever

from src.detection.detector import HallucinationDetector

from src.evaluation.metrics import HallucinationMetrics


st.set_page_config(
    page_title="Hallucination Detection using RAG",
    layout="wide"
)

st.title(
    "Hallucination Detection and Correction in LLMs using RAG"
)


query = st.text_input(
    "Enter your question:"
)


if st.button("Analyze") and query:

    with st.spinner(
        "Running pipeline..."
    ):

        generator = BaseAnswerGenerator()

        raw_answer = generator.generate_answer(
            query
        )

        retriever = SemanticRetriever()

        retrieval = retriever.retrieve(
            query
        )

        evidence = retrieval[
            "documents"
        ][0]

        detector = HallucinationDetector()

        raw_results = detector.detect(
            raw_answer,
            evidence
        )

        corrector = AnswerCorrector()

        corrected_answer = corrector.correct(
            query,
            evidence
        )

        corrected_results = detector.detect(
            corrected_answer,
            evidence
        )

        before_support = (
            HallucinationMetrics.compute_support_ratio(
                raw_results
            )
        )

        after_support = (
            HallucinationMetrics.compute_support_ratio(
                corrected_results
            )
        )

        hallucination_rate = (
            HallucinationMetrics.compute_hallucination_rate(
                raw_results
            )
        )

        improvement = (
            HallucinationMetrics.compute_improvement(
                before_support,
                after_support
            )
        )

    st.header(
        "Raw LLM Answer"
    )

    st.write(
        raw_answer
    )

    st.header(
        "Retrieved Evidence"
    )

    for i, ev in enumerate(
        evidence
    ):

        st.write(
            f"{i+1}. {ev}"
        )

    st.header(
        "Claim Analysis"
    )

    for row in raw_results:

        st.write(
            f"{row['label']} | "
            f"{row['score']} | "
            f"{row['claim']}"
        )

    st.header(
        "Corrected Answer"
    )

    st.success(
        corrected_answer
    )

    st.header(
        "Evaluation"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Before Support",
        before_support
    )

    col2.metric(
        "After Support",
        after_support
    )

    col3.metric(
        "Hallucination Rate",
        hallucination_rate
    )

    col4.metric(
        "Improvement",
        improvement
    )