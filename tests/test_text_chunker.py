from voice_ai_bot.openai_voice import TextChunker


def test_chunker_splits_on_sentence_boundary():
    chunker = TextChunker(target_chars=20)

    chunks = chunker.push("Hello there. Next sentence")

    assert chunks == ["Hello there."]
    assert chunker.flush() == "Next sentence"


def test_chunker_splits_long_text_without_sentence_boundary():
    chunker = TextChunker(target_chars=10)

    chunks = chunker.push("one two three four")

    assert chunks == ["one two"]
    assert chunker.flush() == "three four"
