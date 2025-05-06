import textstat
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_sbert_features(sbert_model, claim, evidence):
    """Extract Sentence-BERT embeddings for claim and evidence."""
    claim_embedding = sbert_model.encode(claim)
    evidence_embedding = sbert_model.encode(evidence)
    return {
        "sbert_cosine_similarity": cosine_similarity(
            [claim_embedding], [evidence_embedding]
        )[0][0],
        "sbert_claim_embedding": claim_embedding,
        "sbert_evidence_embedding": evidence_embedding,
    }


def extract_linguistic_features(spacy_model, claim, evidence):
    """Extract linguistic features from claim and evidence pair."""
    features = {}

    # Basic length features
    features["claim_length"] = len(claim)
    features["evidence_length"] = len(evidence)
    features["length_ratio"] = features["claim_length"] / max(
        features["evidence_length"], 1
    )

    # Tokenization
    claim_tokens = word_tokenize(claim.lower())
    evidence_tokens = word_tokenize(evidence.lower())

    # Lexical features
    features["claim_token_count"] = len(claim_tokens)
    features["evidence_token_count"] = len(evidence_tokens)
    features["claim_unique_tokens"] = len(set(claim_tokens))
    features["evidence_unique_tokens"] = len(set(evidence_tokens))

    # Lexical richness
    features["claim_lexical_density"] = features["claim_unique_tokens"] / max(
        features["claim_token_count"], 1
    )
    features["evidence_lexical_density"] = features["evidence_unique_tokens"] / max(
        features["evidence_token_count"], 1
    )

    # Readability scores
    features["claim_flesch_reading_ease"] = textstat.flesch_reading_ease(claim)
    features["evidence_flesch_reading_ease"] = textstat.flesch_reading_ease(evidence)
    features["claim_flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(claim)
    features["evidence_flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(evidence)

    # Overlap features
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    overlap = claim_set.intersection(evidence_set)

    features["token_overlap"] = len(overlap)
    features["token_overlap_ratio"] = len(overlap) / max(
        len(claim_set.union(evidence_set)), 1
    )

    # Named entity overlap
    claim_doc = spacy_model(claim)
    evidence_doc = spacy_model(evidence)

    claim_entities = set([ent.text.lower() for ent in claim_doc.ents])
    evidence_entities = set([ent.text.lower() for ent in evidence_doc.ents])

    entity_overlap = claim_entities.intersection(evidence_entities)
    features["entity_overlap"] = len(entity_overlap)
    features["entity_overlap_ratio"] = (
        len(entity_overlap) / max(len(claim_entities.union(evidence_entities)), 1)
        if claim_entities or evidence_entities
        else 0
    )

    # Hedge words and certainty markers
    hedge_words = set(
        [
            "may",
            "might",
            "could",
            "suggest",
            "indicate",
            "possible",
            "potential",
            "likely",
            "probably",
        ]
    )
    certainty_words = set(
        [
            "prove",
            "confirm",
            "demonstrate",
            "establish",
            "verify",
            "definitely",
            "certainly",
            "clearly",
            "indeed",
        ]
    )

    claim_text_lower = claim.lower()
    evidence_text_lower = evidence.lower()

    features["claim_hedge_count"] = sum(
        1 for word in hedge_words if word in claim_text_lower
    )
    features["evidence_hedge_count"] = sum(
        1 for word in hedge_words if word in evidence_text_lower
    )
    features["claim_certainty_count"] = sum(
        1 for word in certainty_words if word in claim_text_lower
    )
    features["evidence_certainty_count"] = sum(
        1 for word in certainty_words if word in evidence_text_lower
    )

    # Negation features
    negation_words = set(
        ["not", "no", "never", "n't", "none", "neither", "nor", "without"]
    )
    features["claim_negation_count"] = sum(
        1 for word in negation_words if word in claim_text_lower
    )
    features["evidence_negation_count"] = sum(
        1 for word in negation_words if word in evidence_text_lower
    )

    # Dependency parsing features (subject-verb-object alignment)
    claim_svo = extract_svo(claim_doc)
    evidence_svo = extract_svo(evidence_doc)

    flat_claim_svo = [item for sublist in claim_svo for item in sublist]
    flat_evidence_svo = [item for sublist in evidence_svo for item in sublist]

    # Calculate SVO overlap
    svo_overlap = set(flat_claim_svo).intersection(set(flat_evidence_svo))
    flat_svo_overlap = set(flat_claim_svo).intersection(set(flat_evidence_svo))
    features["svo_overlap"] = len(svo_overlap)
    features["flat_svo_overlap"] = len(flat_svo_overlap)

    return features


def extract_semantic_features(spacy_model, claim, evidence, ngram_range=(1, 2)):
    """Extract semantic features using TF-IDF and semantic similarity."""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform([claim, evidence])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Process with spaCy for additional semantic analysis
    claim_doc = spacy_model(claim)
    evidence_doc = spacy_model(evidence)

    # SpaCy document similarity (vector similarity)
    spacy_similarity = claim_doc.similarity(evidence_doc)

    return {"tfidf_cosine_similarity": cosine_sim, "spacy_similarity": spacy_similarity}


def extract_svo(doc):
    """Extract subject-verb-object triples from a spaCy doc."""
    return [
        (subj.text.lower(), token.text.lower(), obj.text.lower())
        for token in doc
        if token.pos_ == "VERB"
        for subj in token.lefts
        if subj.dep_ in {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        for obj in token.rights
        if obj.dep_ in {"dobj", "dative", "attr", "oprd"}
    ]
