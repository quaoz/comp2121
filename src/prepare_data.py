import json

import numpy as np
import pandas as pd


def load_jsonl(file_path):
    """Load jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_claim_evidence_pairs(claims, corpus_dict):
    """
    Extract claim-evidence pairs for classification.

    For each claim, create:
    - Positive examples: claim + evidence sentences that support/contradict the claim
    - Negative examples: claim + randomly selected sentences from cited documents that are not evidence
    """
    data = []

    for claim_obj in claims:
        # skip claims lacking evidence
        if "evidence" not in claim_obj or not claim_obj["evidence"]:
            continue

        # process evidence for this claim
        for doc_id, rationales in claim_obj["evidence"].items():
            doc_id = int(doc_id)
            doc = corpus_dict[doc_id]
            abstract_sentences = doc["abstract"]

            for rationale in rationales:
                evidence_sentences = [
                    abstract_sentences[i] for i in rationale["sentences"]
                ]
                evidence_text = " ".join(evidence_sentences)

                # create positive example
                data.append(
                    {
                        "claim_id": claim_obj["id"],
                        "doc_id": doc_id,
                        "claim": claim_obj["claim"],
                        "title": doc["title"],
                        "evidence": evidence_text,
                        "label": rationale["label"],
                        "is_evidence": 1,
                    }
                )

                # get sentence indices not in this evidence set
                non_evidence_indices = [
                    i
                    for i in range(len(abstract_sentences))
                    if i not in rationale["sentences"]
                ]

                # create negative examples (only if we have non-evidence sentences)
                if non_evidence_indices:
                    # select a similar number of random non-evidence sentences
                    num_to_sample = min(
                        len(rationale["sentences"]), len(non_evidence_indices)
                    )
                    sampled_indices = np.random.choice(
                        non_evidence_indices, num_to_sample, replace=False
                    )

                    non_evidence_sentences = [
                        abstract_sentences[i] for i in sampled_indices
                    ]
                    non_evidence_text = " ".join(non_evidence_sentences)

                    data.append(
                        {
                            "claim_id": claim_obj["id"],
                            "doc_id": doc_id,
                            "claim": claim_obj["claim"],
                            "title": doc["title"],
                            "evidence": non_evidence_text,
                            "label": "NO_RELATION",
                            "is_evidence": 0,
                        }
                    )

    return pd.DataFrame(data)
