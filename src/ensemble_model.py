import numpy as np
import torch


class EnsembleModel:
    """
    Ensemble model that combines traditional ML models with transformer predictions.
    Uses a weighted voting approach to combine predictions.
    """

    def __init__(
        self,
        device,
        binary_classifier,
        relation_classifier,
        bert_model,
        tokenizer,
        meta_classifier,
    ):
        self.device = device
        self.binary_classifier = binary_classifier
        self.relation_classifier = relation_classifier
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        self.bert_model.eval()
        self.bert_model.to(self.device)

        self.meta_classifier = meta_classifier

    def train_meta_classifier(self, combined_features, labels):
        """Train the meta-classifier to combine traditional and transformer predictions."""
        self.meta_classifier.fit(combined_features, labels)

    def get_bert_predictions(self, claims, evidences, batch_size=16):
        """Get predictions from the transformer model."""
        inputs = [
            claim + " [SEP] " + evidence for claim, evidence in zip(claims, evidences)
        ]
        all_probs = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            encoded_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded_inputs["input_ids"].to(self.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict(self, features_df, claims, evidences):
        """
        Make predictions using the ensemble.

        Args:
            features_df: DataFrame containing traditional features
            claims: List of claim texts
            evidences: List of evidence texts

        Returns:
            Predictions with three classes: 'SUPPORT', 'CONTRADICT', 'NO_RELATION'
        """
        # Get traditional model probabilities
        feature_columns = [
            col
            for col in features_df.columns
            if col not in ["claim_id", "doc_id", "label", "is_evidence"]
        ]
        binary_probs = self.binary_classifier.predict_proba(
            features_df[feature_columns]
        )
        relation_probs = np.zeros(
            (len(features_df), 2)
        )  # Default probabilities for all samples
        is_evidence_mask = features_df["is_evidence"] == 1
        relation_probs[is_evidence_mask] = self.relation_classifier.predict_proba(
            features_df[feature_columns][is_evidence_mask]
        )

        # Get transformer model probabilities
        bert_probs = self.get_bert_predictions(claims, evidences)

        # Combine probabilities into a feature matrix
        combined_probs = np.hstack([binary_probs, relation_probs, bert_probs])

        # Make predictions using the meta-classifier
        final_predictions = self.meta_classifier.predict(combined_probs)

        return final_predictions
