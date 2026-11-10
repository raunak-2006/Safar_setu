import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app as app_module
from app import app
from app import DESTINATION_AREAS
from multilingual_translator import (
    SUPPORTED_LANGUAGES,
    MultilingualTranslator,
    TranslationUnavailableError,
)
from phrase_translator import PhraseTranslator


ROOT = Path(__file__).resolve().parents[1]


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_phrasebook_translation_is_case_and_punctuation_insensitive(self):
        response = self.client.post(
            "/api/translate",
            json={
                "text": "WHERE IS THE RAILWAY STATION?",
                "source_language": "en",
                "target_language": "hi",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["translation"], "रेलवे स्टेशन कहाँ है")
        self.assertEqual(response.json["method"], "phrasebook")

    def test_blank_phrase_is_rejected(self):
        response = self.client.post("/api/translate", json={"text": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Enter a phrase to translate.")

    def test_non_object_request_is_rejected(self):
        response = self.client.post("/api/translate", json=["not", "an", "object"])
        self.assertEqual(response.status_code, 400)

    def test_supported_languages_are_returned(self):
        response = self.client.get("/api/languages")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {language["code"] for language in response.json},
            set(SUPPORTED_LANGUAGES),
        )
        self.assertEqual(len(response.json), 41)

    def test_unknown_language_is_rejected(self):
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Where is the station?",
                "source_language": "en",
                "target_language": "xx",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_unhashable_language_code_is_rejected(self):
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Where is the station?",
                "source_language": "en",
                "target_language": ["ta"],
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_selected_language_is_passed_to_local_model(self):
        with patch.object(
            app_module.multilingual_translator,
            "translate",
            return_value="ரயில் நிலையம் எங்கே?",
        ) as translate:
            response = self.client.post(
                "/api/translate",
                json={
                    "text": "Where is the railway station?",
                    "source_language": "en",
                    "target_language": "ta",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["translation"], "ரயில் நிலையம் எங்கே?")
        self.assertEqual(response.json["language"], "ta")
        self.assertEqual(response.json["method"], "local_ai")
        translate.assert_called_once_with(
            "Where is the railway station?",
            SUPPORTED_LANGUAGES["en"],
            SUPPORTED_LANGUAGES["ta"],
        )

    def test_api_rejects_non_english_source(self):
        response = self.client.post(
            "/api/translate",
            json={
                "text": "नमस्ते",
                "source_language": "hi",
                "target_language": "en",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "SafarSetu accepts English input only.")

    def test_unavailable_local_model_returns_actionable_error(self):
        with patch.object(
            app_module.multilingual_translator,
            "translate",
            side_effect=TranslationUnavailableError("Install local AI dependencies."),
        ):
            response = self.client.post(
                "/api/translate",
                json={
                    "text": "Where is the railway station?",
                    "source_language": "en",
                    "target_language": "ta",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json)

    def test_overlong_phrase_is_rejected(self):
        response = self.client.post(
            "/api/translate",
            json={"text": "a" * 501, "target_language": "hi"},
        )
        self.assertEqual(response.status_code, 400)

    def test_api_rejects_english_as_target_language(self):
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Hello there!",
                "source_language": "en",
                "target_language": "en",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Choose a regional target language.")

    def test_local_model_uses_selected_target_language(self):
        class FakeEncoding(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            unk_token_id = 0
            src_lang = None

            def convert_tokens_to_ids(self, code):
                self.requested_code = code
                return 42

            def __call__(self, text, **kwargs):
                return FakeEncoding(input_ids=[[1, 2]])

            def batch_decode(self, generated, **kwargs):
                return ["ரயில் நிலையம் எங்கே?"]

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                self.generation_options = kwargs
                return [[1, 2, 3]]

        translator = MultilingualTranslator()
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator._tokenizer = tokenizer
        translator._model = model
        translator._torch = type(
            "FakeTorch",
            (),
            {"inference_mode": staticmethod(nullcontext)},
        )

        result = translator.translate(
            "Where is the railway station?",
            SUPPORTED_LANGUAGES["en"],
            SUPPORTED_LANGUAGES["ta"],
        )

        self.assertEqual(result, "ரயில் நிலையம் எங்கே?")
        self.assertEqual(tokenizer.src_lang, SUPPORTED_LANGUAGES["en"].nllb_code)
        self.assertEqual(tokenizer.requested_code, "tam_Taml")
        self.assertEqual(model.generation_options["forced_bos_token_id"], 42)

    def test_unmatched_phrase_has_explicit_fallback_message(self):
        translator = PhraseTranslator(ROOT / "data" / "travel_phrases.csv", ROOT / "missing-artifacts")
        result = translator.translate("Can you recommend a museum?")
        self.assertEqual(result["translation"], "")
        self.assertEqual(result["method"], "unavailable")

    def test_partial_model_artifacts_raise_an_explicit_error(self):
        with TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            (artifacts_dir / "attention_model.keras").touch()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete model artifacts"):
                PhraseTranslator(ROOT / "data" / "travel_phrases.csv", artifacts_dir)

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SafarSetu", response.data)
        self.assertIn(b'id="target-language"', response.data)
        self.assertIn(b"Tamil", response.data)
        self.assertIn(b'for="target-search"', response.data)
        self.assertNotIn(b'id="source-language"', response.data)
        self.assertNotIn(b'id="swap-languages"', response.data)
        self.assertIn(b'id="voice-assistant"', response.data)
        self.assertIn(b"browser speech recognition needs internet", response.data)
        self.assertIn(b'id="destination-area"', response.data)
        self.assertIn(b'id="emergency-support"', response.data)
        self.assertIn(b'data-map-service="police station"', response.data)
        self.assertIn(b'data-map-service="hospital"', response.data)
        self.assertIn(b'id="emergency-note"', response.data)
        self.assertIn(b"brand-tagline", response.data)
        self.assertEqual(len(DESTINATION_AREAS), 36)
        self.assertNotIn(b'value="en"', response.data)
        self.assertTrue(
            all(area["language_code"] in SUPPORTED_LANGUAGES for area in DESTINATION_AREAS)
        )
        self.assertIn(b'Chinese (Simplified)', response.data)
        self.assertIn(b'Portuguese', response.data)


if __name__ == "__main__":
    unittest.main()
