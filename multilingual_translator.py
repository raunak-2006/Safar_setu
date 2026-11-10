from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native_name: str
    nllb_code: str
    speech_code: str
    category: str


SUPPORTED_LANGUAGES = {
    language.code: language
    for language in (
        Language("hi", "Hindi", "हिन्दी", "hin_Deva", "hi-IN", "Indian languages"),
        Language("bn", "Bengali", "বাংলা", "ben_Beng", "bn-IN", "Indian languages"),
        Language("ta", "Tamil", "தமிழ்", "tam_Taml", "ta-IN", "Indian languages"),
        Language("te", "Telugu", "తెలుగు", "tel_Telu", "te-IN", "Indian languages"),
        Language("mr", "Marathi", "मराठी", "mar_Deva", "mr-IN", "Indian languages"),
        Language("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", "gu-IN", "Indian languages"),
        Language("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", "kn-IN", "Indian languages"),
        Language("ml", "Malayalam", "മലയാളം", "mal_Mlym", "ml-IN", "Indian languages"),
        Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru", "pa-IN", "Indian languages"),
        Language("ur", "Urdu", "اردو", "urd_Arab", "ur-IN", "Indian languages"),
        Language("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya", "or-IN", "Indian languages"),
        Language("as", "Assamese", "অসমীয়া", "asm_Beng", "as-IN", "Indian languages"),
        Language("sa", "Sanskrit", "संस्कृतम्", "san_Deva", "sa-IN", "Indian languages"),
        Language("ks", "Kashmiri", "कॉशुर", "kas_Arab", "ks-IN", "Indian languages"),
        Language("sd", "Sindhi", "سنڌي", "snd_Arab", "sd-IN", "Indian languages"),
        Language("bho", "Bhojpuri", "भोजपुरी", "bho_Deva", "bho-IN", "Indian languages"),
        Language("mai", "Maithili", "मैथिली", "mai_Deva", "mai-IN", "Indian languages"),
        Language("awa", "Awadhi", "अवधी", "awa_Deva", "awa-IN", "Indian languages"),
        Language("hne", "Chhattisgarhi", "छत्तीसगढ़ी", "hne_Deva", "hne-IN", "Indian languages"),
        Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "sat_Beng", "sat-IN", "Indian languages"),
        Language("mni", "Manipuri", "মৈতৈলোন্", "mni_Beng", "mni-IN", "Indian languages"),
        Language("lus", "Mizo", "Mizo", "lus_Latn", "lus-IN", "Indian languages"),
        Language("mag", "Magahi", "मगही", "mag_Deva", "mag-IN", "Indian languages"),
        Language("ne", "Nepali", "नेपाली", "npi_Deva", "ne-NP", "Indian languages"),
        Language("si", "Sinhala", "සිංහල", "sin_Sinh", "si-LK", "South & East Asia"),
        Language("ps", "Pashto", "پښتو", "pbt_Arab", "ps-AF", "South & East Asia"),
        Language("my", "Burmese", "မြန်မာ", "mya_Mymr", "my-MM", "South & East Asia"),
        Language("th", "Thai", "ไทย", "tha_Thai", "th-TH", "South & East Asia"),
        Language("ms", "Malay", "Bahasa Melayu", "zsm_Latn", "ms-MY", "South & East Asia"),
        Language("id", "Indonesian", "Bahasa Indonesia", "ind_Latn", "id-ID", "South & East Asia"),
        Language("ar", "Arabic", "العربية", "arb_Arab", "ar-SA", "World languages"),
        Language("fa", "Persian", "فارسی", "pes_Arab", "fa-IR", "World languages"),
        Language("zh", "Chinese (Simplified)", "简体中文", "zho_Hans", "zh-CN", "World languages"),
        Language("ja", "Japanese", "日本語", "jpn_Jpan", "ja-JP", "World languages"),
        Language("ko", "Korean", "한국어", "kor_Hang", "ko-KR", "World languages"),
        Language("fr", "French", "Français", "fra_Latn", "fr-FR", "World languages"),
        Language("es", "Spanish", "Español", "spa_Latn", "es-ES", "World languages"),
        Language("de", "German", "Deutsch", "deu_Latn", "de-DE", "World languages"),
        Language("ru", "Russian", "Русский", "rus_Cyrl", "ru-RU", "World languages"),
        Language("pt", "Portuguese", "Português", "por_Latn", "pt-PT", "World languages"),
        Language("en", "English", "English", "eng_Latn", "en-US", "World languages"),
    )
}

MODEL_ID = "facebook/nllb-200-distilled-600M"
MAX_INPUT_CHARACTERS = 500


class TranslationUnavailableError(RuntimeError):
    """Raised when the local translation model cannot be used."""


class MultilingualTranslator:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._lock = RLock()

    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
    ) -> str:
        if len(text) > MAX_INPUT_CHARACTERS:
            raise ValueError(
                f"Keep phrases under {MAX_INPUT_CHARACTERS} characters for best results."
            )

        with self._lock:
            self._ensure_model()
            self._tokenizer.src_lang = source_language.nllb_code
            target_id = self._tokenizer.convert_tokens_to_ids(target_language.nllb_code)
            if target_id == self._tokenizer.unk_token_id:
                raise TranslationUnavailableError(
                    f"The local model does not include {target_language.name}."
                )

            try:
                encoded = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self._model.device)
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **encoded,
                        forced_bos_token_id=target_id,
                        max_new_tokens=128,
                        num_beams=4,
                    )
                translated = self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )[0].strip()
            except (OSError, RuntimeError) as error:
                raise TranslationUnavailableError(
                    "The local translation model could not finish this request. "
                    "Check that the model fits in available memory and try again."
                ) from error

            if not translated:
                raise TranslationUnavailableError(
                    "The local model did not produce a translation. Please rephrase."
                )
            return translated

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise TranslationUnavailableError(
                "Local AI translation needs PyTorch and Transformers. "
                "Install them with: pip install -r requirements.txt"
            ) from error

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
        except (OSError, RuntimeError) as error:
            raise TranslationUnavailableError(
                f"Could not download or load {self.model_id}. Check your internet "
                "connection and available memory; the model is downloaded on first use."
            ) from error

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
