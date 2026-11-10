import csv
import re
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


class PhraseTranslator:
    def __init__(
        self,
        dataset_path: Path,
        artifacts_dir: Path,
        load_model: bool = True,
    ):
        self.phrasebook: dict[str, str] = {}
        with dataset_path.open(encoding="utf-8", newline="") as dataset_file:
            for row in csv.DictReader(dataset_file):
                english = row["english"].strip()
                hindi = row["hindi"].strip()
                if english and hindi:
                    self.phrasebook[normalize(english)] = hindi

        self.model: Any = None
        self.source_tokenizer: Any = None
        self.target_tokenizer: Any = None
        model_path = artifacts_dir / "attention_model.keras"
        source_tokenizer_path = artifacts_dir / "source_tokenizer.json"
        target_tokenizer_path = artifacts_dir / "target_tokenizer.json"
        artifact_paths = (
            model_path,
            source_tokenizer_path,
            target_tokenizer_path,
        )
        present_paths = [path for path in artifact_paths if path.exists()]
        if load_model and present_paths and len(present_paths) != len(artifact_paths):
            missing = [path.name for path in artifact_paths if not path.exists()]
            raise FileNotFoundError(
                f"Incomplete model artifacts in {artifacts_dir}; missing: {', '.join(missing)}"
            )

        if load_model and len(present_paths) == len(artifact_paths):
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.text import tokenizer_from_json

            self.model = load_model(model_path)
            self.source_tokenizer = tokenizer_from_json(
                source_tokenizer_path.read_text(encoding="utf-8")
            )
            self.target_tokenizer = tokenizer_from_json(
                target_tokenizer_path.read_text(encoding="utf-8")
            )

    def translate(self, text: str) -> dict[str, str]:
        cleaned = text.strip()
        phrase_match = self.phrasebook.get(normalize(cleaned))
        if phrase_match:
            return {
                "translation": phrase_match,
                "method": "phrasebook",
                "message": "Exact phrase-book match.",
            }

        if self.model is None:
            return {
                "translation": "",
                "method": "unavailable",
                "message": "This phrase is not in the phrase book. Add it to the dataset or train the model.",
            }

        source_words = normalize(cleaned).split()
        if any(word not in self.source_tokenizer.word_index for word in source_words):
            return {
                "translation": "",
                "method": "out_of_vocabulary",
                "message": "This phrase contains words outside the trained vocabulary and is not in the phrase book.",
            }

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        source_sequence = self.source_tokenizer.texts_to_sequences([cleaned.casefold()])
        source_length = self.model.input_shape[0][1]
        source = pad_sequences(source_sequence, maxlen=source_length, padding="post")

        start_id = self.target_tokenizer.word_index.get("<start>")
        end_id = self.target_tokenizer.word_index.get("<end>")
        if start_id is None or end_id is None:
            raise ValueError("The target tokenizer is missing the start/end tokens.")

        generated = [start_id]
        for _ in range(self.model.input_shape[1][1]):
            decoder = pad_sequences(
                [generated],
                maxlen=self.model.input_shape[1][1],
                padding="post",
            )
            probabilities = self.model.predict([source, decoder], verbose=0)
            next_id = int(probabilities[0, len(generated) - 1].argmax())
            if next_id in (0, start_id, end_id):
                break
            generated.append(next_id)

        index_word = self.target_tokenizer.index_word
        translation = " ".join(index_word.get(token_id, "") for token_id in generated[1:])
        if not translation:
            return {
                "translation": "",
                "method": "model",
                "message": "The model did not produce a translation. Use a phrase-book entry instead.",
            }
        return {
            "translation": translation,
            "method": "model",
            "message": "Experimental model output; verify important or emergency translations.",
        }


def load_training_data(dataset_path: Path) -> list[tuple[str, str]]:
    with dataset_path.open(encoding="utf-8", newline="") as dataset_file:
        rows = csv.DictReader(dataset_file)
        return [
            (row["english"].strip(), row["hindi"].strip())
            for row in rows
            if row["english"].strip() and row["hindi"].strip()
        ]
