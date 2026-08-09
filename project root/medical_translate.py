"""Medical Translate — minimal proof-of-concept.
Microphone -> Whisper -> NLLB -> Piper
"""

from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from piper.voice import PiperVoice
import ctranslate2, sounddevice as sd, numpy as np

app = Flask(__name__)

# Example local models
whisper = WhisperModel("models/whisper", device="cpu", compute_type="int8")
tokenizer = AutoTokenizer.from_pretrained("models/nllb")
nllb = ctranslate2.Translator("models/nllb", device="cpu")
voice = PiperVoice.load("models/piper/en.onnx")

LANG = {
    "en": "eng_Latn", "fa": "pes_Arab", "ar": "arb_Arab",
    "tr": "tur_Latn", "ru": "rus_Cyrl"
}


def transcribe(audio):
    segments, info = whisper.transcribe(audio, vad_filter=True)
    return " ".join(x.text.strip() for x in segments), info.language


def translate(text, src, dst):
    tokenizer.src_lang = LANG[src]
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    result = nllb.translate_batch([tokens], target_prefix=[[LANG[dst]]])
    ids = tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
    return tokenizer.decode(ids, skip_special_tokens=True)


@app.get("/")
def home():
    return render_template("medical_translate_github.html")


@app.post("/translate")
def run_translation():
    target = request.form.get("target", "en")
    audio = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="float32")
    sd.wait()

    text, source = transcribe(audio.squeeze())
    output = text if source == target else translate(text, source, target)
    return jsonify(speech=text, translation=output, language=target)


@app.post("/speak")
def speak():
    text = request.json.get("text", "")
    with sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype="int16") as stream:
        for chunk in voice.synthesize(text):
            stream.write(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
    return ""


if __name__ == "__main__":
    app.run()
