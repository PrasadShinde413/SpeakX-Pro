import nltk
import textstat
import re

# Download required NLTK data on first run
def _ensure_nltk_data():
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_data()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def _vocabulary_richness(text):
    """
    Type-Token Ratio (TTR): unique words / total words.
    Score range: 0–1 (higher = more varied vocabulary).
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.0, 0, 0
    total = len(words)
    unique = len(set(words))
    ttr = round(unique / total, 3)
    return ttr, unique, total


def _coherence_score(text):
    """
    Sentence-level coherence via word overlap between consecutive sentences.
    Score range: 0–1 (higher = more cohesive).
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0, len(sentences)

    stop_words = set(stopwords.words('english'))

    def meaningful_words(sentence):
        words = set(re.findall(r'\b[a-zA-Z]+\b', sentence.lower()))
        return words - stop_words

    overlaps = []
    for i in range(1, len(sentences)):
        prev = meaningful_words(sentences[i - 1])
        curr = meaningful_words(sentences[i])
        if prev and curr:
            overlap = len(prev & curr) / len(prev | curr)
            overlaps.append(overlap)

    coherence = round(float(sum(overlaps) / len(overlaps)), 3) if overlaps else 0.0
    return coherence, len(sentences)


def _sentence_complexity(text):
    """
    Compute sentence complexity metrics using textstat.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return {}

    words_per_sentence = [
        len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences
    ]
    avg_sentence_length = round(
        sum(words_per_sentence) / len(words_per_sentence), 1
    ) if words_per_sentence else 0

    flesch_score = round(textstat.flesch_reading_ease(text), 1)
    flesch_grade = round(textstat.flesch_kincaid_grade(text), 1)

    # Flesch Reading Ease interpretation
    if flesch_score >= 80:
        readability_label = "Very Easy"
    elif flesch_score >= 60:
        readability_label = "Standard"
    elif flesch_score >= 40:
        readability_label = "Fairly Difficult"
    else:
        readability_label = "Difficult"

    return {
        "avg_sentence_length": avg_sentence_length,
        "flesch_reading_ease": flesch_score,
        "flesch_kincaid_grade": flesch_grade,
        "readability": readability_label,
        "sentence_count": len(sentences)
    }


def _grammar_check(text):
    """
    Grammar check using language_tool_python.
    Tries local Java server first, then falls back to Public API.
    Returns error count, top error categories, and detailed match list.
    """
    try:
        import language_tool_python

        # Try local Java LanguageTool first
        try:
            tool = language_tool_python.LanguageTool('en-US')
        except Exception:
            # Fallback: use the free public API (no Java required)
            tool = language_tool_python.LanguageToolPublicAPI('en-US')

        matches = tool.check(text)
        tool.close()

        error_count = len(matches)

        # Summarize error categories
        categories = {}
        for m in matches:
            cat = m.ruleId.split('_')[0]
            categories[cat] = categories.get(cat, 0) + 1

        # Top 3 categories
        top_errors = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]

        # Detailed error list for the dropdown report
        detailed_errors = []
        for m in matches[:15]:  # Limit to first 15 to avoid info overload
            detailed_errors.append({
                "message": m.message,
                "context": m.context,
                "suggestions": m.replacements[:3] if m.replacements else [],
                "rule": m.ruleId
            })

        return error_count, [f"{k}({v})" for k, v in top_errors], detailed_errors

    except Exception as e:
        print(f"Grammar check skipped: {e}")
        return -1, ["Grammar check unavailable"], []


def analyze_nlp(transcript):
    """
    Run all NLP analysis on the provided transcript text.
    Returns a dictionary of NLP metrics.
    """
    if not transcript or transcript.strip() == "":
        return {
            "grammar_errors": 0,
            "grammar_top_issues": [],
            "vocabulary_ttr": 0.0,
            "unique_words": 0,
            "total_words": 0,
            "coherence_score": 0.0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "readability": "N/A"
        }

    print("Running NLP analysis...")

    # Grammar
    grammar_errors, grammar_top_issues, grammar_errors_detail = _grammar_check(transcript)

    # Vocabulary richness
    ttr, unique_words, total_words = _vocabulary_richness(transcript)

    # Coherence
    coherence, sentence_count = _coherence_score(transcript)

    # Sentence complexity
    complexity = _sentence_complexity(transcript)

    return {
        # Grammar
        "grammar_errors": grammar_errors,
        "grammar_top_issues": grammar_top_issues,
        "grammar_errors_detail": grammar_errors_detail,

        # Vocabulary
        "vocabulary_ttr": ttr,
        "unique_words": unique_words,
        "total_words": total_words,

        # Coherence
        "coherence_score": coherence,
        "sentence_count": sentence_count,

        # Sentence Complexity
        "avg_sentence_length": complexity.get("avg_sentence_length", 0),
        "flesch_reading_ease": complexity.get("flesch_reading_ease", 0),
        "flesch_kincaid_grade": complexity.get("flesch_kincaid_grade", 0),
        "readability": complexity.get("readability", "N/A")
    }
