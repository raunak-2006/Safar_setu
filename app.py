import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from multilingual_translator import (
    MAX_INPUT_CHARACTERS,
    SUPPORTED_LANGUAGES,
    MultilingualTranslator,
    TranslationUnavailableError,
)
from phrase_translator import PhraseTranslator, load_training_data


ROOT = Path(__file__).resolve().parent
PHRASEBOOK_CATEGORIES = (
    "Essentials",
    "Directions & transport",
    "Hotel & stays",
    "Food & dining",
    "Medical",
    "Police & safety",
)
DESTINATION_AREAS = (
    {"name": "Andhra Pradesh", "language_code": "te"},
    {"name": "Arunachal Pradesh", "language_code": "hi"},
    {"name": "Assam", "language_code": "as"},
    {"name": "Bihar", "language_code": "hi"},
    {"name": "Chhattisgarh", "language_code": "hne"},
    {"name": "Goa", "language_code": "mr"},
    {"name": "Gujarat", "language_code": "gu"},
    {"name": "Haryana", "language_code": "hi"},
    {"name": "Himachal Pradesh", "language_code": "hi"},
    {"name": "Jharkhand", "language_code": "hi"},
    {"name": "Karnataka", "language_code": "kn"},
    {"name": "Kerala", "language_code": "ml"},
    {"name": "Madhya Pradesh", "language_code": "hi"},
    {"name": "Maharashtra", "language_code": "mr"},
    {"name": "Manipur", "language_code": "mni"},
    {"name": "Meghalaya", "language_code": "hi"},
    {"name": "Mizoram", "language_code": "lus"},
    {"name": "Nagaland", "language_code": "hi"},
    {"name": "Odisha", "language_code": "or"},
    {"name": "Punjab", "language_code": "pa"},
    {"name": "Rajasthan", "language_code": "hi"},
    {"name": "Sikkim", "language_code": "ne"},
    {"name": "Tamil Nadu", "language_code": "ta"},
    {"name": "Telangana", "language_code": "te"},
    {"name": "Tripura", "language_code": "bn"},
    {"name": "Uttar Pradesh", "language_code": "hi"},
    {"name": "Uttarakhand", "language_code": "hi"},
    {"name": "West Bengal", "language_code": "bn"},
    {"name": "Andaman and Nicobar Islands", "language_code": "hi"},
    {"name": "Chandigarh", "language_code": "pa"},
    {"name": "Dadra and Nagar Haveli and Daman and Diu", "language_code": "gu"},
    {"name": "Delhi", "language_code": "hi"},
    {"name": "Jammu and Kashmir", "language_code": "ur"},
    {"name": "Ladakh", "language_code": "ur"},
    {"name": "Lakshadweep", "language_code": "ml"},
    {"name": "Puducherry", "language_code": "ta"},
)
app = Flask(__name__)
translator = PhraseTranslator(
    ROOT / "data" / "travel_phrases.csv",
    ROOT / "artifacts",
    load_model=False,
)
multilingual_translator = MultilingualTranslator()


def phrase_category(english: str) -> str:
    phrase = english.casefold()
    if any(word in phrase for word in ("room", "reservation", "booking", "breakfast")):
        return "Hotel & stays"
    if any(
        word in phrase
        for word in ("menu", "water", "vegetarian", "food", "spicy", "allergic", "bill")
    ):
        return "Food & dining"
    if any(word in phrase for word in ("hospital", "ambulance", "doctor", "medicine", "hurt")):
        return "Medical"
    if any(
        word in phrase
        for word in ("police", "help", "lost", "stolen", "safe", "passport", "family")
    ):
        return "Police & safety"
    if any(
        word in phrase
        for word in (
            "station",
            "airport",
            "bus",
            "train",
            "platform",
            "ticket",
            "taxi",
            "fare",
            "address",
            "map",
            "far",
            "left",
            "right",
            "straight",
            "way",
            "city centre",
            "stop here",
        )
    ):
        return "Directions & transport"
    return "Essentials"


def load_offline_phrasebooks() -> dict[str, dict[str, list[dict[str, str]]]]:
    phrasebooks: dict[str, dict[str, list[dict[str, str]]]] = {
        "hi": {
            "language_code": "hi",
            "entries": [
                {
                    "english": english,
                    "translation": hindi,
                    "category": phrase_category(english),
                }
                for english, hindi in load_training_data(
                    ROOT / "data" / "travel_phrases.csv"
                )
            ],
        }
    }

    pack_directory = ROOT / "data" / "phrasebooks"
    if not pack_directory.exists():
        return phrasebooks

    for pack_path in sorted(pack_directory.glob("*.json")):
        language_code = pack_path.stem
        if language_code not in SUPPORTED_LANGUAGES or language_code == "en":
            raise ValueError(
                f"Offline phrasebook {pack_path.name} must use a supported target-language code."
            )
        try:
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid offline phrasebook JSON: {pack_path}") from error
        if not isinstance(pack, dict) or not isinstance(pack.get("entries"), list):
            raise ValueError(f"Offline phrasebook {pack_path} must contain an entries list.")

        entries = []
        seen_phrases = set()
        for entry in pack["entries"]:
            if not isinstance(entry, dict):
                raise ValueError(f"Every phrase in {pack_path} must be an object.")
            english = entry.get("english")
            translation = entry.get("translation")
            category = entry.get("category")
            if (
                not isinstance(english, str)
                or not english.strip()
                or not isinstance(translation, str)
                or not translation.strip()
                or category not in PHRASEBOOK_CATEGORIES
            ):
                raise ValueError(
                    f"Every phrase in {pack_path} needs English, translation, and a valid category."
                )
            normalized = english.strip().casefold()
            if normalized in seen_phrases:
                raise ValueError(f"Duplicate English phrase in {pack_path}: {english}")
            seen_phrases.add(normalized)
            entries.append(
                {
                    "english": english.strip(),
                    "translation": translation.strip(),
                    "category": category,
                }
            )

        phrasebooks[language_code] = {
            "language_code": language_code,
            "entries": entries,
        }
    return phrasebooks


offline_phrasebooks = load_offline_phrasebooks()


@app.get("/")
def home():
    return render_template(
        "index.html",
        languages=[
            {
                "code": language.code,
                "name": language.name,
                "native_name": language.native_name,
                "speech_code": language.speech_code,
                "category": language.category,
            }
            for language in SUPPORTED_LANGUAGES.values()
        ],
        destination_areas=DESTINATION_AREAS,
        offline_phrasebooks=offline_phrasebooks,
        phrasebook_categories=PHRASEBOOK_CATEGORIES,
    )


@app.get("/api/languages")
def languages():
    return jsonify(
        [
            {
                "code": language.code,
                "name": language.name,
                "native_name": language.native_name,
                "speech_code": language.speech_code,
                "category": language.category,
            }
            for language in SUPPORTED_LANGUAGES.values()
        ]
    )


@app.post("/api/translate")
def translate():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter a phrase to translate."}), 400
    if len(text.strip()) > MAX_INPUT_CHARACTERS:
        return jsonify(
            {
                "error": (
                    f"Keep phrases under {MAX_INPUT_CHARACTERS} characters "
                    "for best results."
                )
            }
        ), 400

    target_code = payload.get("target_language")
    source_code = payload.get("source_language", "en")
    if source_code != "en":
        return jsonify({"error": "SafarSetu accepts English input only."}), 400
    if not isinstance(target_code, str) or target_code not in SUPPORTED_LANGUAGES:
        return jsonify({"error": "Choose a supported target language."}), 400
    if target_code == "en":
        return jsonify({"error": "Choose a regional target language."}), 400
    source_language = SUPPORTED_LANGUAGES["en"]
    target_language = SUPPORTED_LANGUAGES[target_code]

    if source_language.code == "en" and target_language.code == "hi":
        phrasebook_result = translator.translate(text)
        if phrasebook_result["method"] == "phrasebook":
            return jsonify(
                {
                    **phrasebook_result,
                    "language": target_language.code,
                    "language_name": target_language.name,
                    "speech_code": target_language.speech_code,
                }
            )

    try:
        translated_text = multilingual_translator.translate(
            text.strip(),
            source_language,
            target_language,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except TranslationUnavailableError as error:
        return jsonify({"error": str(error)}), 503

    return jsonify(
        {
            "translation": translated_text,
            "method": "local_ai",
            "message": (
                "Local AI model translation. Check important translations with "
                "a fluent speaker."
            ),
            "language": target_language.code,
            "language_name": target_language.name,
            "speech_code": target_language.speech_code,
            "source_language": source_language.code,
        }
    )


if __name__ == "__main__":
    app.run()
