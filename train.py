import argparse
import json
import random
from pathlib import Path

import numpy as np
import sacrebleu
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from phrase_translator import load_training_data


def make_model(
    source_vocab_size: int,
    target_vocab_size: int,
    source_length: int,
    target_length: int,
    use_attention: bool,
):
    source_input = keras.Input(shape=(source_length,), name="source")
    target_input = keras.Input(shape=(target_length,), name="target")

    source_embedding = keras.layers.Embedding(
        source_vocab_size, 128, name="source_embedding"
    )(source_input)
    encoder_sequence, encoder_state = keras.layers.GRU(
        128, return_sequences=True, return_state=True, name="encoder"
    )(
        source_embedding,
        mask=keras.ops.not_equal(source_input, 0),
    )

    target_embedding = keras.layers.Embedding(
        target_vocab_size, 128, name="target_embedding"
    )(target_input)
    decoder_sequence, _ = keras.layers.GRU(
        128, return_sequences=True, return_state=True, name="decoder"
    )(
        target_embedding,
        initial_state=encoder_state,
        mask=keras.ops.not_equal(target_input, 0),
    )

    if use_attention:
        context = keras.layers.AdditiveAttention(name="bahdanau_attention")(
            [decoder_sequence, encoder_sequence],
            mask=[
                keras.ops.not_equal(target_input, 0),
                keras.ops.not_equal(source_input, 0),
            ],
        )
        decoder_sequence = keras.layers.Concatenate(name="context_concat")(
            [decoder_sequence, context]
        )

    probabilities = keras.layers.Dense(
        target_vocab_size, activation="softmax", name="token_probabilities"
    )(decoder_sequence)
    model = keras.Model([source_input, target_input], probabilities)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )
    return model


def tokenize_pairs(pairs: list[tuple[str, str]]):
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    source_tokenizer = Tokenizer(filters=filters, lower=True)
    target_tokenizer = Tokenizer(filters=filters, lower=True)
    sources = [source for source, _ in pairs]
    targets = [f"<start> {target} <end>" for _, target in pairs]
    source_tokenizer.fit_on_texts(sources)
    target_tokenizer.fit_on_texts(targets)

    source_sequences = source_tokenizer.texts_to_sequences(sources)
    target_sequences = target_tokenizer.texts_to_sequences(targets)
    source_length = max(map(len, source_sequences))
    target_length = max(map(len, target_sequences))
    source = pad_sequences(source_sequences, maxlen=source_length, padding="post")
    target = pad_sequences(target_sequences, maxlen=target_length, padding="post")
    return source_tokenizer, target_tokenizer, source, target, source_length, target_length


def greedy_decode(model, source, target_tokenizer, source_length, target_length):
    start_id = target_tokenizer.word_index["<start>"]
    end_id = target_tokenizer.word_index["<end>"]
    translations = []
    for source_row in source:
        generated = [start_id]
        for _ in range(target_length - 1):
            decoder = pad_sequences(
                [generated], maxlen=target_length, padding="post"
            )
            probabilities = model.predict(
                [source_row.reshape(1, source_length), decoder], verbose=0
            )
            next_id = int(probabilities[0, len(generated) - 1].argmax())
            if next_id in (0, start_id, end_id):
                break
            generated.append(next_id)
        translations.append(
            " ".join(target_tokenizer.index_word.get(token_id, "") for token_id in generated[1:])
        )
    return translations


def main():
    parser = argparse.ArgumentParser(description="Train and compare seq2seq translation models.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=Path("data/travel_phrases.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    pairs = load_training_data(args.data)
    if len(pairs) < 10:
        raise ValueError("Provide at least 10 English-Hindi phrase pairs to train and evaluate.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    validation_size = max(1, round(len(indices) * 0.2))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    source_tokenizer, target_tokenizer, source, target, source_length, target_length = (
        tokenize_pairs(pairs)
    )
    train_source, validation_source = source[training_indices], source[validation_indices]
    train_target, validation_target = target[training_indices], target[validation_indices]
    train_decoder = train_target[:, :-1]
    train_labels = train_target[:, 1:, np.newaxis]
    train_weights = (train_labels[..., 0] != 0).astype(np.float32)
    validation_decoder = validation_target[:, :-1]
    validation_labels = validation_target[:, 1:, np.newaxis]
    validation_weights = (validation_labels[..., 0] != 0).astype(np.float32)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, use_attention in (("baseline", False), ("attention", True)):
        print(f"Training {name} model...")
        model = make_model(
            len(source_tokenizer.word_index) + 1,
            len(target_tokenizer.word_index) + 1,
            source_length,
            target_length - 1,
            use_attention,
        )
        model.fit(
            [train_source, train_decoder],
            train_labels,
            sample_weight=train_weights,
            validation_data=(
                [validation_source, validation_decoder],
                validation_labels,
                validation_weights,
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
        )
        hypotheses = greedy_decode(
            model, validation_source, target_tokenizer, source_length, target_length - 1
        )
        references = []
        for row in validation_target:
            special_tokens = {
                0,
                target_tokenizer.word_index["<start>"],
                target_tokenizer.word_index["<end>"],
            }
            tokens = [
                target_tokenizer.index_word.get(int(token_id), "")
                for token_id in row
                if token_id not in special_tokens
            ]
            references.append(" ".join(tokens))
        scores[name] = sacrebleu.corpus_bleu(hypotheses, [references]).score
        if use_attention:
            model.save(args.artifacts / "attention_model.keras")

    (args.artifacts / "evaluation.json").write_text(
        json.dumps(
            {
                "validation_pairs": len(validation_indices),
                "bleu": scores,
                "note": "BLEU on this small random holdout is only a development signal, not a measure of real-world safety.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.artifacts / "source_tokenizer.json").write_text(
        source_tokenizer.to_json(), encoding="utf-8"
    )
    (args.artifacts / "target_tokenizer.json").write_text(
        target_tokenizer.to_json(), encoding="utf-8"
    )
    print(f"BLEU scores: {scores}")
    print(f"Saved attention model and tokenizers in {args.artifacts}")


if __name__ == "__main__":
    main()
