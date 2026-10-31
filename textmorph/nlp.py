import math
import re
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textmorph.config import PARAPHRASE_MODEL, SIMILARITY_MODEL, SUMMARY_MODELS

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model_cache = {}
_tokenizer_cache = {}
_embedder = None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Please enter text or upload a supported document.")
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_gutenberg(text: str) -> str:
    start = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS)?\s*.*?PROJECT GUTENBERG.*?\*\*\*",
        text,
        flags=re.I | re.S,
    )
    end = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS)?\s*.*?PROJECT GUTENBERG.*",
        text,
        flags=re.I | re.S,
    )
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text.strip()


def validate_text(text: str, minimum: int = 1) -> str:
    text = clean_text(text)
    if len(text) < minimum:
        raise ValueError(f"Please provide at least {minimum} characters.")
    if not re.search(r"\w", text, flags=re.UNICODE):
        raise ValueError("Please provide text containing words or numbers.")
    return text


def split_sentences(text: str, minimum_words: int = 1, maximum_chars: int = 4000) -> List[str]:
    text = clean_text(text)
    fragments = re.split(r"(?<=[.!?])\s+(?=[\"'“‘(]*[\w])|\n+", text)
    sentences = []
    for fragment in fragments:
        sentence = re.sub(r"\s+", " ", fragment).strip()
        if not sentence or len(sentence) > maximum_chars:
            continue
        if re.match(r"^(chapter|book|part|section)\s+(?:[ivxlcdm]+|\d+)\b", sentence, re.I):
            continue
        if len(re.findall(r"\b[\w'-]+\b", sentence, flags=re.UNICODE)) < minimum_words:
            continue
        sentences.append(sentence)
    return sentences


def summarization_preprocess(text: str) -> str:
    return validate_text(strip_gutenberg(text), minimum=40)


def paraphrasing_preprocess(text: str) -> str:
    return validate_text(strip_gutenberg(text), minimum=1)


def summarization_sentences(text: str) -> List[str]:
    return split_sentences(text, minimum_words=1, maximum_chars=4000)


def paraphrasing_sentences(text: str) -> List[str]:
    return split_sentences(text, minimum_words=1, maximum_chars=4000)


def similarity_sentences(text: str) -> List[str]:
    return split_sentences(clean_text(text), minimum_words=1, maximum_chars=4000)


def _max_input_tokens(tokenizer, fallback: int = 1024) -> int:
    configured = getattr(tokenizer, "model_max_length", fallback)
    if not isinstance(configured, int) or configured > 100_000:
        return fallback
    return min(configured, fallback)


def _token_ids(text: str, tokenizer) -> List[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def make_chunks(text: str, tokenizer, limit: int = 850, overlap: int = 0) -> List[str]:
    if limit < 1:
        raise ValueError("The token limit must be positive.")
    sentences = summarization_sentences(text)
    if not sentences:
        sentences = [clean_text(text)]
    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if len(_token_ids(candidate, tokenizer)) <= limit:
            current.append(sentence)
            continue
        if current:
            chunks.append(" ".join(current))
            if overlap:
                tail = current[-1:]
                while tail and len(_token_ids(" ".join(tail + [sentence]), tokenizer)) > limit:
                    tail = tail[1:]
                current = tail
            else:
                current = []
        sentence_ids = _token_ids(sentence, tokenizer)
        if len(sentence_ids) > limit:
            if current:
                chunks.append(" ".join(current))
                current = []
            for start in range(0, len(sentence_ids), max(1, limit - overlap)):
                part = tokenizer.decode(sentence_ids[start:start + limit], skip_special_tokens=True).strip()
                if part:
                    chunks.append(part)
            current = []
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks or [clean_text(text)]


def load_seq2seq(model_id: str):
    if model_id not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(DEVICE)
        model.eval()
        _tokenizer_cache[model_id] = tokenizer
        _model_cache[model_id] = model
    return _tokenizer_cache[model_id], _model_cache[model_id]


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(SIMILARITY_MODEL, device=DEVICE)
    return _embedder


def _summarize_one(text: str, model_id: str, tokenizer, model, max_words: int) -> str:
    prompt = f"summarize: {text}" if model_id == "t5-small" else text
    token_limit = _max_input_tokens(tokenizer, fallback=1024 if "bart" in model_id.lower() else 512)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    if inputs["input_ids"].shape[-1] > token_limit:
        raise ValueError("An internal summary chunk exceeded the selected model's context limit.")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_words),
            min_new_tokens=min(25, max(1, int(max_words) // 3)),
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.8,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def summarize(text: str, model_label: str, max_words: int = 130) -> str:
    text = summarization_preprocess(text)
    if model_label not in SUMMARY_MODELS:
        raise ValueError("Choose a supported summarization model.")
    if not 16 <= int(max_words) <= 512:
        raise ValueError("Summary length must be between 16 and 512 tokens.")
    model_id = SUMMARY_MODELS[model_label]
    tokenizer, model = load_seq2seq(model_id)
    context_limit = _max_input_tokens(tokenizer, fallback=1024 if "bart" in model_id.lower() else 512)
    prefix_size = len(_token_ids("summarize:", tokenizer)) if model_id == "t5-small" else 0
    chunks = make_chunks(text, tokenizer, limit=max(1, context_limit - prefix_size - 4))
    summaries = [_summarize_one(chunk, model_id, tokenizer, model, max_words) for chunk in chunks]
    summaries = [summary for summary in summaries if summary]
    if not summaries:
        return "No summary was generated. Try a longer input."

    for _ in range(6):
        combined = "\n".join(summaries)
        prompt = f"summarize: {combined}" if model_id == "t5-small" else combined
        if len(_token_ids(prompt, tokenizer)) <= context_limit:
            return _summarize_one(combined, model_id, tokenizer, model, max_words) if len(summaries) > 1 else summaries[0]
        reduction_chunks = make_chunks(combined, tokenizer, limit=max(1, context_limit - prefix_size - 4))
        next_summaries = [_summarize_one(chunk, model_id, tokenizer, model, max_words) for chunk in reduction_chunks]
        next_summaries = [summary for summary in next_summaries if summary]
        if not next_summaries or next_summaries == summaries:
            break
        summaries = next_summaries
    raise ValueError("The document is too long to consolidate safely with this model. Use a smaller document or a model with a larger context window.")


_STYLE_PROMPTS = {
    "Standard": "paraphrase: {sentence}",
    "Formal": "paraphrase in a formal and professional style: {sentence}",
    "Simple English": "paraphrase in clear, simple English: {sentence}",
    "Academic": "paraphrase in an academic style: {sentence}",
    "Creative": "paraphrase with expressive wording while preserving the meaning: {sentence}",
}


def paraphrase_one(sentence: str, count: int = 3, style: str = "Standard") -> List[str]:
    sentence = validate_text(sentence, minimum=1)
    if style not in _STYLE_PROMPTS:
        raise ValueError("Choose a supported paraphrasing style.")
    tokenizer, model = load_seq2seq(PARAPHRASE_MODEL)
    prompt = _STYLE_PROMPTS[style].format(sentence=sentence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    n = max(1, min(int(count), 5))
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=max(6, n),
            num_return_sequences=n,
            no_repeat_ngram_size=2,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    seen, result = set(), []
    original = re.sub(r"\W+", "", sentence.casefold(), flags=re.UNICODE)
    for item in candidates:
        item = item.strip()
        key = re.sub(r"\W+", "", item.casefold(), flags=re.UNICODE)
        if item and key != original and key not in seen:
            seen.add(key)
            result.append(item)
    return result or ["No distinct paraphrase was generated. Try a complete sentence."]


def paraphrase(text: str, count: int = 3, max_sentences: int = 5, style: str = "Standard") -> str:
    text = paraphrasing_preprocess(text)
    sentences = paraphrasing_sentences(text) or [text]
    sentences = sentences[:max(1, min(int(max_sentences), 8))]
    sections = []
    for index, sentence in enumerate(sentences, 1):
        variants = paraphrase_one(sentence, count, style)
        sections.append(
            f"### Sentence {index}\n**Original:** {sentence}\n\n" +
            "\n".join(f"{i}. {item}" for i, item in enumerate(variants, 1))
        )
    return "\n\n---\n\n".join(sections)


def similarity_score(text_a: str, text_b: str) -> float:
    a = validate_text(text_a, minimum=3)
    b = validate_text(text_b, minimum=3)
    vectors = get_embedder().encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])


def similarity(text_a: str, text_b: str) -> str:
    score = similarity_score(text_a, text_b)
    if score >= .80:
        meaning = "Very high semantic similarity"
    elif score >= .60:
        meaning = "Moderate-to-high semantic similarity"
    elif score >= .40:
        meaning = "Moderate semantic similarity"
    else:
        meaning = "Low semantic similarity"
    return f"## {score:.3f}\n{meaning}\n\n*Cosine similarity measures semantic closeness, not factual equivalence.*"


def similarity_analysis(text_a: str, text_b: str):
    a = validate_text(text_a, minimum=3)
    b = validate_text(text_b, minimum=3)
    sentences_a, sentences_b = similarity_sentences(a), similarity_sentences(b)
    if not sentences_a:
        sentences_a = [a]
    if not sentences_b:
        sentences_b = [b]
    if len(sentences_a) > 25 or len(sentences_b) > 25:
        raise ValueError("Sentence-level comparison is limited to 25 sentences per input.")
    embedder = get_embedder()
    vectors = embedder.encode(
        sentences_a + sentences_b,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    left = vectors[:len(sentences_a)]
    right = vectors[len(sentences_a):]
    matrix = cosine_similarity(left, right)
    document_score = float(cosine_similarity([vectors[:len(sentences_a)].mean(axis=0)], [vectors[len(sentences_a):].mean(axis=0)])[0][0])
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("The similarity heatmap requires Matplotlib. Install the optional requirements.") from error
    figure, axis = plt.subplots(figsize=(max(5, len(sentences_b) * .55), max(4, len(sentences_a) * .42)))
    image = axis.imshow(matrix, cmap="YlOrBr", vmin=-1, vmax=1, aspect="auto")
    axis.set_xlabel("Text B sentence")
    axis.set_ylabel("Text A sentence")
    axis.set_xticks(range(len(sentences_b)), labels=range(1, len(sentences_b) + 1))
    axis.set_yticks(range(len(sentences_a)), labels=range(1, len(sentences_a) + 1))
    axis.set_title("Sentence-level cosine similarity")
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    return document_score, matrix, figure


def text_statistics(text: str) -> Dict[str, float]:
    text = clean_text(text)
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    sentences = split_sentences(text)
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    return {
        "Words": len(words),
        "Characters": len(text),
        "Characters (excluding spaces)": sum(not char.isspace() for char in text),
        "Sentences": len(sentences),
        "Paragraphs": len(paragraphs),
        "Average sentence length (words)": round(len(words) / max(1, len(sentences)), 2),
        "Estimated reading time (minutes)": math.ceil(len(words) / 220) if words else 0,
    }


def _syllable_count(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 1
    groups = re.findall(r"[aeiouy]+", word)
    count = max(1, len(groups))
    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1
    return count


def readability(text: str) -> Dict[str, object]:
    text = validate_text(text, minimum=1)
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    sentences = split_sentences(text)
    if not words or not sentences:
        return {"Status": "Not enough text to estimate readability."}

    def heuristic_result():
        syllables = sum(_syllable_count(word) for word in words)
        word_count, sentence_count = len(words), len(sentences)
        return {
            "Flesch Reading Ease": round(206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count), 2),
            "Flesch-Kincaid Grade Level": round(0.39 * (word_count / sentence_count) + 11.8 * (syllables / word_count) - 15.59, 2),
            "Average sentence length (words)": round(word_count / sentence_count, 2),
            "Average word length (characters)": round(sum(map(len, words)) / word_count, 2),
            "Method": "Approximate English syllable heuristic",
        }

    try:
        import textstat
        return {
            "Flesch Reading Ease": round(float(textstat.flesch_reading_ease(text)), 2),
            "Flesch-Kincaid Grade Level": round(float(textstat.flesch_kincaid_grade(text)), 2),
            "Average sentence length (words)": round(len(words) / len(sentences), 2),
            "Average word length (characters)": round(sum(map(len, words)) / len(words), 2),
            "Method": "textstat; designed primarily for English",
        }
    except ImportError:
        return heuristic_result()
    except (ValueError, ZeroDivisionError):
        return heuristic_result()


def rouge_eval(generated: str, reference: str) -> str:
    from rouge_score import rouge_scorer
    generated = validate_text(generated, minimum=3)
    reference = validate_text(reference, minimum=3)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    rows = ["| Metric | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    rows.extend(
        f"| {name.upper()} | {score.precision:.4f} | {score.recall:.4f} | {score.fmeasure:.4f} |"
        for name, score in scores.items()
    )
    return "\n".join(rows)


def summarize_timed(text: str, model_label: str, max_words: int = 130) -> Tuple[str, float]:
    started = time.perf_counter()
    result = summarize(text, model_label, max_words)
    return result, time.perf_counter() - started
