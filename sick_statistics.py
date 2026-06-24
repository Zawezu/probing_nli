import json
from functools import lru_cache

from transformers import AutoTokenizer

from utils import MERGED_SICK_FILEPATH, LABEL_MAP, MODELS_FOLDER

# Tokenizer of the Tiny Aya Global model (see activations.py), used to count tokens
# consistently across all languages.
_TOKENIZER_MODEL_NAME = "tiny_aya_global"


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(
        f"{MODELS_FOLDER}/{_TOKENIZER_MODEL_NAME}", local_files_only=True
    )


@lru_cache(maxsize=1)
def _load_data() -> dict:
    with open(MERGED_SICK_FILEPATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _standard_label_key(language: str) -> str:
    return "standard_japanese_label" if language == "jp" else "standard_label"


def count_samples(language: str, split: str | None = None) -> int:
    """Count samples for a language, optionally restricted to a specific split."""
    data = _load_data()
    sentence_key = f"sentence_a_{language}"
    return sum(
        1
        for entry in data.values()
        if sentence_key in entry and (split is None or entry["split"] == split)
    )


def label_percentage(language: str, label: str) -> float:
    """Percentage of sentence pairs classified with the given label name ('entailment', 'neutral', or 'contradiction')."""
    data = _load_data()
    label_int = LABEL_MAP[label]
    label_key = _standard_label_key(language)
    sentence_key = f"sentence_a_{language}"

    total = 0
    matching = 0
    for entry in data.values():
        if sentence_key not in entry:
            continue
        total += 1
        if entry[label_key] == label_int:
            matching += 1

    return (matching / total * 100) if total > 0 else 0.0


def label_alignment(language: str) -> float:
    """Percentage of pairs whose label matches the English standard label."""
    if language == "en":
        return 100.0

    data = _load_data()
    lang_label_key = _standard_label_key(language)
    sentence_key = f"sentence_a_{language}"

    total = 0
    matching = 0
    for entry in data.values():
        if sentence_key not in entry:
            continue
        total += 1
        if entry[lang_label_key] == entry["standard_label"]:
            matching += 1

    return (matching / total * 100) if total > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    return _tokenizer().tokenize(text)


def _count_tokens(text: str, language: str) -> int:
    return len(_tokenize(text))


def avg_lexical_overlap(language: str, label: str | None = None) -> float:
    """Average Jaccard overlap between premise and hypothesis token sets.

    Overlap per pair = |tokens(premise) ∩ tokens(hypothesis)| / |tokens(premise) ∪ tokens(hypothesis)|,
    computed on lowercased tokens. Optionally restricted to pairs with a specific label.
    """
    data = _load_data()
    label_key = _standard_label_key(language)
    label_int = LABEL_MAP[label] if label is not None else None

    total_overlap = 0.0
    total_samples = 0
    for entry in data.values():
        if f"sentence_a_{language}" not in entry:
            continue
        if label_int is not None and entry[label_key] != label_int:
            continue
        prem_tokens = {t.lower() for t in _tokenize(entry[f"sentence_a_{language}"])}
        hypo_tokens = {t.lower() for t in _tokenize(entry[f"sentence_b_{language}"])}
        union = prem_tokens | hypo_tokens
        if union:
            total_overlap += len(prem_tokens & hypo_tokens) / len(union)
        total_samples += 1

    return (total_overlap / total_samples * 100) if total_samples > 0 else 0.0


def avg_tokens(language: str, part: str, label: str | None = None) -> float:
    """Average number of tokens in the premise or hypothesis.

    Uses the Tiny Aya Global model tokenizer for all languages, so token counts are
    comparable across languages.
    Optionally restricted to pairs with a specific label ('entailment', 'neutral', or 'contradiction').
    """
    data = _load_data()
    sentence_key = (
        f"sentence_a_{language}" if part == "premise" else f"sentence_b_{language}"
    )
    label_key = _standard_label_key(language)
    label_int = LABEL_MAP[label] if label is not None else None

    total_tokens = 0
    total_samples = 0
    for entry in data.values():
        if sentence_key not in entry:
            continue
        if label_int is not None and entry[label_key] != label_int:
            continue
        total_tokens += _count_tokens(entry[sentence_key], language)
        total_samples += 1

    return total_tokens / total_samples if total_samples > 0 else 0.0


def fill_table() -> None:
    """Gather all statistics and print the LaTeX table."""
    langs = ["en", "es", "nl", "jp"]

    total = {lang: count_samples(lang) for lang in langs}
    test = {lang: count_samples(lang, "test") for lang in langs}
    train = {lang: count_samples(lang, "train") for lang in langs}
    val = {lang: count_samples(lang, "val") for lang in langs}

    ent = {lang: label_percentage(lang, "entailment") for lang in langs}
    neu = {lang: label_percentage(lang, "neutral") for lang in langs}
    con = {lang: label_percentage(lang, "contradiction") for lang in langs}

    jp_align = label_alignment("jp")

    overlap = {lang: avg_lexical_overlap(lang) for lang in langs}
    overlap_ent = {lang: avg_lexical_overlap(lang, "entailment") for lang in langs}
    overlap_neu = {lang: avg_lexical_overlap(lang, "neutral") for lang in langs}
    overlap_con = {lang: avg_lexical_overlap(lang, "contradiction") for lang in langs}

    prem = {lang: avg_tokens(lang, "premise") for lang in langs}
    hypo = {lang: avg_tokens(lang, "hypothesis") for lang in langs}

    prem_ent = {lang: avg_tokens(lang, "premise", "entailment") for lang in langs}
    prem_neu = {lang: avg_tokens(lang, "premise", "neutral") for lang in langs}
    prem_con = {lang: avg_tokens(lang, "premise", "contradiction") for lang in langs}
    hypo_ent = {lang: avg_tokens(lang, "hypothesis", "entailment") for lang in langs}
    hypo_neu = {lang: avg_tokens(lang, "hypothesis", "neutral") for lang in langs}
    hypo_con = {lang: avg_tokens(lang, "hypothesis", "contradiction") for lang in langs}

    def p(v: float) -> str:
        return f"{v:.1f}"

    en, es, nl, jp = "en", "es", "nl", "jp"

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(
        r"\caption{Descriptive statistics for the four versions of the SICK dataset. Lexical overlap is the average Jaccard similarity ($|A \cap B| / |A \cup B|$) between the lowercased token sets of each premise-hypothesis pair.}"
    )
    print(r"\label{tab:dataset-statistics}")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(
        r"\textbf{Statistic} & \textbf{SICK (EN)} & \textbf{SICK-ES} & \textbf{SICK-NL} & \textbf{JSICK} \\ \midrule"
    )
    print(
        rf"\textbf{{Total Size}} & {total[en]} & {total[es]} & {total[nl]} & {total[jp]} \\"
    )
    print(rf"Train Set Size & {train[en]} & {train[es]} & {train[nl]} & {train[jp]} \\")
    print(
        rf"Test Set Size & {test[en]} & {test[es]} & {test[nl]} & {test[jp]} \\ \midrule"
    )
    print(rf"Validation Set Size & {val[en]} & {val[es]} & {val[nl]} & {val[jp]} \\")
    print(r"\textbf{Label Ratios} & & & & \\")
    print(
        rf"Entailment & {p(ent[en])}\% & {p(ent[es])}\% & {p(ent[nl])}\% & {p(ent[jp])}\% \\"
    )
    print(
        rf"Neutral & {p(neu[en])}\% & {p(neu[es])}\% & {p(neu[nl])}\% & {p(neu[jp])}\% \\"
    )
    print(
        rf"Contradiction & {p(con[en])}\% & {p(con[es])}\% & {p(con[nl])}\% & {p(con[jp])}\% \\"
    )
    print(
        rf"\textbf{{Label Alignment}} & 100\% & 100\% & 100\% & {p(jp_align)}\% \\ \midrule"
    )
    print(r"(vs. English labels) & & & & \\ \midrule")
    print(r"\textbf{Lexical Overlap} & & & & \\")
    print(
        rf"Avg. Lexical Overlap & {p(overlap[en])}\% & {p(overlap[es])}\% & {p(overlap[nl])}\% & {p(overlap[jp])}\% \\"
    )
    print(
        rf"\quad Entailment & {p(overlap_ent[en])}\% & {p(overlap_ent[es])}\% & {p(overlap_ent[nl])}\% & {p(overlap_ent[jp])}\% \\"
    )
    print(
        rf"\quad Neutral & {p(overlap_neu[en])}\% & {p(overlap_neu[es])}\% & {p(overlap_neu[nl])}\% & {p(overlap_neu[jp])}\% \\"
    )
    print(
        rf"\quad Contradiction & {p(overlap_con[en])}\% & {p(overlap_con[es])}\% & {p(overlap_con[nl])}\% & {p(overlap_con[jp])}\% \\ \midrule"
    )
    print(r"\textbf{Token Counts} & & & & \\")
    print(
        rf"Avg. Tokens (Premise) & {p(prem[en])} & {p(prem[es])} & {p(prem[nl])} & {p(prem[jp])} \\"
    )
    print(
        rf"\quad Entailment & {p(prem_ent[en])} & {p(prem_ent[es])} & {p(prem_ent[nl])} & {p(prem_ent[jp])} \\"
    )
    print(
        rf"\quad Neutral & {p(prem_neu[en])} & {p(prem_neu[es])} & {p(prem_neu[nl])} & {p(prem_neu[jp])} \\"
    )
    print(
        rf"\quad Contradiction & {p(prem_con[en])} & {p(prem_con[es])} & {p(prem_con[nl])} & {p(prem_con[jp])} \\"
    )
    print(
        rf"Avg. Tokens (Hypothesis) & {p(hypo[en])} & {p(hypo[es])} & {p(hypo[nl])} & {p(hypo[jp])} \\"
    )
    print(
        rf"\quad Entailment & {p(hypo_ent[en])} & {p(hypo_ent[es])} & {p(hypo_ent[nl])} & {p(hypo_ent[jp])} \\"
    )
    print(
        rf"\quad Neutral & {p(hypo_neu[en])} & {p(hypo_neu[es])} & {p(hypo_neu[nl])} & {p(hypo_neu[jp])} \\"
    )
    print(
        rf"\quad Contradiction & {p(hypo_con[en])} & {p(hypo_con[es])} & {p(hypo_con[nl])} & {p(hypo_con[jp])} \\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    fill_table()
