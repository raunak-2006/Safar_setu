const form = document.querySelector("#translate-form");
const phraseInput = document.querySelector("#phrase");
const languageSelect = document.querySelector("#target-language");
const areaSelect = document.querySelector("#destination-area");
const areaHint = document.querySelector("#area-hint");
const emergencyNote = document.querySelector("#emergency-note");
const selectedLanguage = document.querySelector("#selected-language");
const characterCount = document.querySelector("#character-count");
const errorMessage = document.querySelector("#error");
const result = document.querySelector("#result");
const translation = document.querySelector("#translation");
const method = document.querySelector("#method");
const resultHeading = document.querySelector("#result-heading");
const speakButton = document.querySelector("#speak");
const submitLabel = document.querySelector("#translate-label");
const voiceButton = document.querySelector("#voice-assistant");
const voiceLabel = document.querySelector("#voice-label");
const voiceStatus = document.querySelector("#voice-status");
const offlinePhrasebooks = JSON.parse(
  document.querySelector("#offline-phrasebooks")?.textContent || "{}",
);
const staticDemo = document.body.dataset.staticDemo === "true";
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let speakAfterTranslation = false;
let recognitionHasResult = false;

function normalizePhrase(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, " ").trim().replace(/\s+/g, " ");
}

function updateSelectedLanguage() {
  const targetOption = languageSelect.selectedOptions[0];
  selectedLanguage.textContent = targetOption.dataset.nativeName;
  translation.lang = targetOption.dataset.speechCode;
  resultHeading.textContent = `${targetOption.textContent.split(" — ")[0].trim().toUpperCase()} TRANSLATION`;
}

function filterLanguages(searchInput, select) {
  const query = searchInput.value
    .trim()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
  const currentValue = select.value;

  for (const option of select.options) {
    const category = option.parentElement.label || "";
    const searchableText = `${option.textContent} ${option.dataset.nativeName} ${category}`
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase();
    option.hidden = Boolean(query) && !searchableText.includes(query);
  }

  select.value = currentValue;
}

function updateVoiceButton(listening) {
  isListening = listening;
  voiceButton.setAttribute("aria-pressed", String(listening));
  voiceButton.classList.toggle("is-listening", listening);
  voiceLabel.textContent = listening ? "Listening…" : "Speak";
  voiceButton.setAttribute(
    "aria-label",
    listening ? "Stop voice input" : "Speak a phrase in English",
  );
}

function updatePhrase(text) {
  phraseInput.value = text.slice(0, phraseInput.maxLength);
  characterCount.textContent = `${phraseInput.value.length} / ${phraseInput.maxLength}`;
}

function updateEmergencyLinks() {
  const area = areaSelect.value || "India";
  const searchArea = areaSelect.value ? `near ${area}` : "in India";

  document.querySelectorAll("[data-map-service]").forEach((link) => {
    const service = link.dataset.mapService;
    const query = `${service} ${searchArea}`;
    link.href = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
  });

  emergencyNote.textContent = areaSelect.value
    ? `Map searches are focused on ${area}. Results are provided by Google Maps; SafarSetu does not access your device location or contact emergency services.`
    : "Choose a destination area above for more focused map searches. Results are provided by Google Maps; SafarSetu does not access your device location or contact emergency services.";
}

function speakCurrentTranslation(automatic = false) {
  if (!("speechSynthesis" in window)) {
    errorMessage.textContent = "Spoken translation is not available in this browser.";
    errorMessage.hidden = false;
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(translation.textContent);
  utterance.lang = translation.lang;
  utterance.onend = () => {
    if (automatic) {
      voiceStatus.textContent = "Translation spoken. Tap Speak to translate another phrase.";
    }
  };
  utterance.onerror = (event) => {
    if (event.error !== "canceled" && event.error !== "interrupted") {
      errorMessage.textContent = "Could not play the spoken translation. Use Listen to try again.";
      errorMessage.hidden = false;
      if (automatic) {
        voiceStatus.textContent = "Translation is ready. Tap Listen to hear it.";
      }
    }
  };
  window.speechSynthesis.speak(utterance);
}

languageSelect.addEventListener("change", updateSelectedLanguage);
updateSelectedLanguage();

if (staticDemo) {
  document.querySelector("#static-demo-notice").hidden = false;
  document.querySelector(".privacy-note").textContent = "Hindi phrasebook runs in this page";
}

areaSelect.addEventListener("change", () => {
  updateEmergencyLinks();
  const selectedArea = areaSelect.selectedOptions[0];
  if (!selectedArea.dataset.language) {
    areaHint.textContent = "Choose a destination area for a language suggestion, or select any language directly. Local languages can vary within an area.";
    return;
  }

  languageSelect.value = selectedArea.dataset.language;
  updateSelectedLanguage();
  const languageName = languageSelect.selectedOptions[0].textContent.split(" — ")[0].trim();
  areaHint.textContent = `${languageName} is suggested for ${selectedArea.textContent}. You can change the language above if needed.`;
});
updateEmergencyLinks();

document.querySelector("#target-search").addEventListener("input", (event) => {
  filterLanguages(event.currentTarget, languageSelect);
});

if (!SpeechRecognition) {
  voiceButton.disabled = true;
  voiceStatus.textContent = "Voice input is not supported in this browser. You can still type a phrase.";
} else {
  voiceButton.addEventListener("click", () => {
    if (isListening) {
      recognition.stop();
      voiceStatus.textContent = "Voice input stopped.";
      return;
    }

    errorMessage.hidden = true;
    result.hidden = true;
    speakAfterTranslation = false;
    recognitionHasResult = false;
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      updateVoiceButton(true);
      voiceStatus.textContent = "Listening… speak now, or tap to stop.";
    };
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      recognitionHasResult = true;
      errorMessage.hidden = true;
      if (!transcript) {
        voiceStatus.textContent = "No words were recognized. Tap Speak and try again.";
        return;
      }

      updatePhrase(transcript);
      voiceStatus.textContent = "Phrase recognized. Translating and speaking the result…";
      speakAfterTranslation = true;
      form.requestSubmit();
    };
    recognition.onerror = (event) => {
      if (recognitionHasResult) {
        return;
      }

      const messages = {
        "no-speech": "No speech was detected. Tap Speak and try again.",
        "not-allowed": "Microphone access was blocked. Allow microphone access in your browser settings.",
        "service-not-allowed": "Your browser's speech service is unavailable or not permitted.",
        "audio-capture": "No microphone was found. Connect a microphone or type your phrase.",
        network: "The browser's speech recognition service could not connect. Check your internet connection; if it keeps failing, type your phrase or try another supported browser.",
      };
      const message = messages[event.error] || "Voice input failed. Please try again or type your phrase.";
      voiceStatus.textContent = message;
      errorMessage.textContent = message;
      errorMessage.hidden = false;
    };
    recognition.onend = () => {
      updateVoiceButton(false);
    };

    try {
      recognition.start();
    } catch (error) {
      updateVoiceButton(false);
      voiceStatus.textContent = "Could not start voice input. Check microphone access and try again.";
      errorMessage.textContent = error.message || voiceStatus.textContent;
      errorMessage.hidden = false;
    }
  });
}

phraseInput.addEventListener("input", () => {
  characterCount.textContent = `${phraseInput.value.length} / ${phraseInput.maxLength}`;
});

document.querySelector("#clear-phrase").addEventListener("click", () => {
  phraseInput.value = "";
  characterCount.textContent = `0 / ${phraseInput.maxLength}`;
  errorMessage.hidden = true;
  result.hidden = true;
  phraseInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  result.hidden = true;
  const submitButton = form.querySelector('button[type="submit"]');
  const requestedByVoice = speakAfterTranslation;
  submitButton.disabled = true;
  if (SpeechRecognition) {
    voiceButton.disabled = true;
  }
  submitLabel.textContent = "Translating…";

  try {
    let data;
    if (staticDemo) {
      const entries = offlinePhrasebooks[languageSelect.value]?.entries || [];
      const normalizedPhrase = normalizePhrase(phraseInput.value);
      const match = entries.find(
        (entry) => normalizePhrase(entry.english) === normalizedPhrase,
      );
      if (!match) {
        throw new Error(
          languageSelect.value === "hi"
            ? "No exact Hindi phrasebook match. Try a phrase from the phrasebook."
            : "GitHub Pages supports exact Hindi phrasebook matches only. Run the Flask app for AI translations.",
        );
      }
      data = {
        translation: match.translation,
        message: "Exact Hindi phrasebook match. Verify important translations with a fluent speaker.",
        speech_code: languageSelect.selectedOptions[0].dataset.speechCode,
      };
    } else {
      const response = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: phraseInput.value,
          source_language: "en",
          target_language: languageSelect.value,
        }),
      });
      data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Unable to translate this phrase.");
      }
    }

    if (!data.translation) {
      speakAfterTranslation = false;
      errorMessage.textContent = data.message;
      errorMessage.hidden = false;
      if (requestedByVoice) {
        voiceStatus.textContent = "Your phrase was recognized, but no translation was returned. Try another phrase or type it in.";
      }
      return;
    }

    errorMessage.hidden = true;
    translation.textContent = data.translation;
    translation.lang = data.speech_code;
    method.textContent = data.message;
    result.hidden = false;
    if (speakAfterTranslation) {
      speakAfterTranslation = false;
      voiceStatus.textContent = "Translation ready. Playing it aloud…";
      speakCurrentTranslation(true);
    }
  } catch (error) {
    speakAfterTranslation = false;
    errorMessage.textContent = error.message || "Could not connect. Please try again.";
    errorMessage.hidden = false;
    if (requestedByVoice) {
      voiceStatus.textContent = "Your phrase was recognized, but translation failed. Please try again.";
    }
  } finally {
    submitButton.disabled = false;
    if (SpeechRecognition) {
      voiceButton.disabled = false;
    }
    submitLabel.textContent = "Translate";
  }
});

document.querySelectorAll(".phrase-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    phraseInput.value = chip.dataset.phrase;
    characterCount.textContent = `${phraseInput.value.length} / ${phraseInput.maxLength}`;
    errorMessage.hidden = true;
    result.hidden = true;
    phraseInput.focus();
  });
});

speakButton.addEventListener("click", () => {
  speakCurrentTranslation();
});
