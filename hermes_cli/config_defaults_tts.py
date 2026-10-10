"""Pure-data streaming speech defaults shared by the configuration tree."""

STREAMING_TTS_DEFAULTS = {
    # Minimum sentence length (chars); shorter fragments batch with the next sentence.
    # 20 suits English; CJK voice setups use ~6. The first emission may override it.
    "min_len": 20,
    # Maximum wait for spoken output after text generation finishes;
    # not a delay before playback. Finite seconds, range 1–600.
    "completion_timeout": 120.0,
    # Override only the first sentence; null inherits min_len for all sentences.
    "first_sentence_min_chars": None,
}
