import re
import spacy


class ClaimExtractor:

    def __init__(self):

        self.nlp = spacy.blank("en")

        self.nlp.add_pipe("sentencizer")

    def clean_claim(
        self,
        text: str
    ):

        text = text.strip()

        # remove markdown
        text = text.replace("*", "")

        # remove bullets / numbering
        text = re.sub(
            r"^\s*[\d\-\.\)]+\s*",
            "",
            text
        )

        # normalize spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def extract_claims(
        self,
        text: str
    ):

        doc = self.nlp(text)

        claims = []

        for sent in doc.sents:

            claim = self.clean_claim(
                sent.text
            )

            if len(claim) < 25:
                continue

            if len(claim.split()) < 5:
                continue

            lower_claim = claim.lower()

            if (
                claim.endswith(":")
                    or "such as:" in lower_claim
                    or "including:" in lower_claim
                    or "consist of:" in lower_claim
                ):
                continue

            claims.append(claim)

        return claims