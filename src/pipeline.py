import re
from typing import Any, Dict, List

from src.config import get_config_value
from src.detection.detector import HallucinationDetector
from src.detection.factual_consistency import extract_capitalized_entities, extract_dates, extract_numbers, extract_years
from src.detection.support_scorer import SupportScorer
from src.evaluation.metrics import CRITICAL_UNSUPPORTED_FLAGS, HallucinationMetrics
from src.generation.base_answer import BaseAnswerGenerator
from src.generation.correction import AnswerCorrector
from src.generation.ollama_client import OllamaClient, OllamaServiceError
from src.logger import get_logger
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import SemanticRetriever
from src.utils.text_cleaning import normalize_for_detection, remove_display_citations


class HallucinationRAGPipeline:
    """End-to-end pipeline for retrieval-grounded hallucination detection."""

    def __init__(self, retrieval_mode: str | None = None) -> None:
        self.logger = get_logger("pipeline")
        shared_embedder = EmbeddingModel()
        shared_ollama_client = OllamaClient()

        mode = (retrieval_mode or get_config_value("settings", "retrieval", "mode", default="dense")).lower()
        if mode == "hybrid":
            self.retriever = HybridRetriever(embedder=shared_embedder)
        else:
            self.retriever = SemanticRetriever(embedder=shared_embedder)

        self.generator = BaseAnswerGenerator(client=shared_ollama_client)
        self.corrector = AnswerCorrector(client=shared_ollama_client)
        self.detector = HallucinationDetector(support_scorer=SupportScorer(embedder=shared_embedder))
        self.retrieval_mode = mode
        self.answerability_threshold = float(
            get_config_value("settings", "retrieval", "answerability_threshold", default=0.45)
        )

    @staticmethod
    def _evidence_strength(item: Dict[str, Any]) -> float:
        # Hybrid final_score is already normalized to 0..1-ish. Prefer it.
        for key in ("final_score", "weighted_score", "dense_similarity", "similarity"):
            value = item.get(key)
            if value is not None:
                try:
                    return max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    pass
        value = item.get("rrf_score")
        try:
            return min(1.0, float(value) * 30.0) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _is_answerable_by_score(cls, evidence: List[Dict[str, Any]], threshold: float) -> bool:
        if not evidence:
            return False
        return max(cls._evidence_strength(item) for item in evidence) >= threshold

    @staticmethod
    def _combined_evidence_text(evidence: List[Dict[str, Any]]) -> str:
        return "\n".join(str(item.get("text", "")) for item in evidence)

    @staticmethod
    def _query_requires_specific_fact(query: str) -> bool:
        clean = normalize_for_detection(query).lower()
        if extract_years(clean) or extract_dates(clean):
            return True
        fact_patterns = (
            "introduced", "invented", "created", "proposed", "released", "published",
            "launched", "developed", "won", "began", "started", "founded", "built",
            "which year", "when", "who", "where", "date", "year",
        )
        return any(pattern in clean for pattern in fact_patterns)

    @staticmethod
    def _specific_fact_supported_by_evidence(query: str, evidence: List[Dict[str, Any]]) -> tuple[bool, str | None]:
        clean_query = normalize_for_detection(query)
        evidence_text = normalize_for_detection(HallucinationRAGPipeline._combined_evidence_text(evidence))
        query_years = set(extract_years(clean_query))
        evidence_years = set(extract_years(evidence_text))
        query_dates = set(extract_dates(clean_query))
        evidence_dates = set(extract_dates(evidence_text))
        query_numbers = set(extract_numbers(clean_query))
        evidence_numbers = set(extract_numbers(evidence_text))

        if query_years and not query_years.issubset(evidence_years):
            return False, "insufficient_evidence_for_specific_fact"
        if query_dates and not query_dates.issubset(evidence_dates):
            return False, "insufficient_evidence_for_specific_fact"
        if query_numbers and HallucinationRAGPipeline._query_requires_specific_fact(clean_query):
            if not query_numbers.issubset(evidence_numbers):
                return False, "insufficient_evidence_for_specific_fact"
        lower = clean_query.lower()
        asks_year = any(token in lower for token in ("when", "which year", "introduced", "released", "published", "launched"))
        if asks_year and not evidence_years and not query_years:
            return False, "insufficient_evidence_for_specific_fact"

        asks_who_intro = any(pattern in lower for pattern in (
            "who introduced", "who invented", "who proposed", "who created", "who developed"
        ))
        if asks_who_intro:
            entities = set(extract_capitalized_entities(evidence_text))
            # Require at least one person/organization-like entity beyond the
            # target acronym/name and at least one relation word in evidence.
            relation_present = any(term in evidence_text.lower() for term in (
                "introduced", "invented", "proposed", "created", "developed", "authored", "written by"
            ))
            stop_entities = {"RAG", "Retrieval", "Augmented", "Generation", "ColBERT", "TruthfulQA"}
            meaningful_entities = {entity for entity in entities if entity not in stop_entities}
            if not relation_present or not meaningful_entities:
                return False, "insufficient_evidence_for_specific_fact"
        return True, None

    @staticmethod
    def _add_evidence_ids(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for index, item in enumerate(evidence, start=1):
            copied = dict(item)
            copied["evidence_id"] = f"Evidence-{index}"
            output.append(copied)
        return output

    @staticmethod
    def _empty_detection() -> Dict[str, Any]:
        return {
            "claims": [],
            "claim_count": 0,
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
            "support_ratio": 0.0,
            "hallucination_rate": 0.0,
            "average_support_score": 0.0,
        }

    @staticmethod
    def _safe_insufficient_response(query: str) -> str:
        query = query.strip()
        if query.endswith('.'):
            query = query[:-1]
        return f'The knowledge base does not provide evidence for the specific claim: "{query}."'

    @staticmethod
    def _claim_has_critical_flags(claim_result: Dict[str, Any]) -> bool:
        flags = set(claim_result.get("rule_flags", []) or [])
        return bool(flags.intersection(CRITICAL_UNSUPPORTED_FLAGS)) or claim_result.get("nli_label") == "contradiction"

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        pieces = re.split(r"(?<=[.!?])\s+", text.strip())
        return [piece.strip() for piece in pieces if piece.strip()]

    @staticmethod
    def _sentence_matches_claim(sentence: str, claim: str) -> bool:
        clean_sentence = normalize_for_detection(sentence).lower()
        clean_claim = normalize_for_detection(claim).lower()
        if not clean_sentence or not clean_claim:
            return False
        if clean_claim in clean_sentence or clean_sentence in clean_claim:
            return True
        claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}|\d+", clean_claim))
        sentence_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}|\d+", clean_sentence))
        if not claim_tokens:
            return False
        return len(claim_tokens & sentence_tokens) / len(claim_tokens) >= 0.65


    @staticmethod
    def _is_rag_query(query: str) -> bool:
        lower = normalize_for_detection(query).lower()
        return any(term in lower for term in ("rag", "retrieval augmented generation", "retrieval-augmented generation"))

    @staticmethod
    def _is_generic_rag_boilerplate(sentence: str) -> bool:
        lower = normalize_for_detection(sentence).lower()
        patterns = (
            "generator uses retrieved passages as context",
            "language model uses retrieved passages as context",
            "generator uses retrieved evidence as context",
        )
        return any(pattern in lower for pattern in patterns)

    def _repair_corrected_answer(self, corrected_answer: str, corrected_detection: Dict[str, Any], query: str = "") -> tuple[str, bool, List[Dict[str, Any]]]:
        critical_claims = [
            claim for claim in corrected_detection.get("claims", [])
            if claim.get("label") == "unsupported"
            and (self._claim_has_critical_flags(claim) or float(claim.get("support_score", 0.0) or 0.0) < 0.40)
        ]
        # For non-RAG queries, remove generic RAG boilerplate even if the claim
        # extractor did not classify it as critical.
        boilerplate_removal = not self._is_rag_query(query)
        if not critical_claims and not boilerplate_removal:
            return corrected_answer, False, []

        sentences = self._split_sentences(corrected_answer)
        kept_sentences: List[str] = []
        removed: List[Dict[str, Any]] = []
        for sentence in sentences:
            matched_critical = None
            for claim in critical_claims:
                if self._sentence_matches_claim(sentence, str(claim.get("claim", ""))):
                    matched_critical = claim
                    break
            if matched_critical:
                removed.append({
                    "sentence": sentence,
                    "claim": matched_critical.get("claim"),
                    "rule_flags": matched_critical.get("rule_flags", []),
                    "highlighted_claim": matched_critical.get("highlighted_claim"),
                })
            elif boilerplate_removal and self._is_generic_rag_boilerplate(sentence):
                removed.append({
                    "sentence": sentence,
                    "claim": sentence,
                    "rule_flags": ["generic_rag_boilerplate_removed"],
                    "highlighted_claim": sentence,
                })
            else:
                kept_sentences.append(sentence)

        if kept_sentences:
            repaired = " ".join(kept_sentences).strip()
        else:
            repaired = "Insufficient evidence available in the knowledge base to provide a supported corrected answer."
        return repaired, True, removed

    def _safe_return(self, query: str, evidence: List[Dict[str, Any]], warnings: List[str], status: str) -> Dict[str, Any]:
        safe_response = self._safe_insufficient_response(query)
        empty_detection = self._empty_detection()
        metrics = HallucinationMetrics.summarize(empty_detection, empty_detection)
        return {
            "query": query,
            "retrieval_mode": self.retrieval_mode,
            "answerable": False,
            "answerability_status": status,
            "warnings": warnings,
            "evidence": evidence,
            "raw_answer": safe_response,
            "raw_detection": empty_detection,
            "corrected_answer": safe_response,
            "corrected_answer_cited": safe_response,
            "used_evidence_ids": [item.get("evidence_id") for item in evidence if item.get("evidence_id")],
            "corrected_detection": empty_detection,
            "correction_repaired": False,
            "removed_unsupported_corrected_claims": [],
            "metrics": metrics,
        }

    def run(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        warnings: List[str] = []
        evidence = self._add_evidence_ids(self.retriever.retrieve(query, top_k=top_k))
        if not evidence:
            warnings.append("No evidence was retrieved. Build the index or expand the knowledge base for better results.")

        answerable_by_score = self._is_answerable_by_score(evidence, self.answerability_threshold)
        specific_supported, specific_warning = self._specific_fact_supported_by_evidence(query, evidence)
        if specific_warning:
            warnings.append(specific_warning)

        if not answerable_by_score:
            if "insufficient_evidence" not in warnings:
                warnings.append("insufficient_evidence")
            return self._safe_return(query, evidence, warnings, "not_answerable")

        if not specific_supported:
            return self._safe_return(query, evidence, warnings, "not_answerable_specific_fact")

        try:
            raw_answer = self.generator.generate_answer(query)
        except OllamaServiceError as exc:
            raw_answer = f"LLM generation failed: {exc}"
            warnings.append("llm_generation_failed")

        raw_detection = self.detector.detect(raw_answer, evidence)
        corrected_answer_cited = self.corrector.correct(query, raw_answer, evidence)
        corrected_answer = remove_display_citations(corrected_answer_cited)
        corrected_detection = self.detector.detect(corrected_answer, evidence)

        repaired_answer, repaired, removed = self._repair_corrected_answer(corrected_answer, corrected_detection, query)
        if repaired:
            corrected_answer = repaired_answer
            corrected_detection = self.detector.detect(corrected_answer, evidence)

        metrics = HallucinationMetrics.summarize(raw_detection, corrected_detection)

        return {
            "query": query,
            "retrieval_mode": self.retrieval_mode,
            "answerable": True,
            "answerability_status": "answerable",
            "warnings": warnings,
            "evidence": evidence,
            "raw_answer": raw_answer,
            "raw_detection": raw_detection,
            "corrected_answer": corrected_answer,
            "corrected_answer_cited": corrected_answer_cited,
            "used_evidence_ids": [item.get("evidence_id") for item in evidence if item.get("evidence_id")],
            "corrected_detection": corrected_detection,
            "correction_repaired": repaired,
            "removed_unsupported_corrected_claims": removed,
            "metrics": metrics,
        }
