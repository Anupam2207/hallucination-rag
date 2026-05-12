import re
import unicodedata
from typing import Iterable


def normalize_whitespace(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_common_noise(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('•', '-')
    text = text.replace('\u00a0', ' ')
    return text


def remove_empty_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    filtered = [line for line in lines if line]
    return '\n'.join(filtered)


def preprocess_text(text: str) -> str:
    text = strip_common_noise(text)
    text = normalize_whitespace(text)
    text = remove_empty_lines(text)
    text = normalize_whitespace(text)
    return text


def preprocess_documents(documents: Iterable[dict]) -> list[dict]:
    processed = []
    for doc in documents:
        cleaned_text = preprocess_text(doc.get('text', ''))
        if not cleaned_text:
            continue
        processed.append({**doc, 'text': cleaned_text, 'text_length': len(cleaned_text)})
    return processed
