from voice_ai_bot.conversation import ConversationStore, Message


def test_store_roundtrip(tmp_path):
    path = tmp_path / "conversation.json"
    store = ConversationStore(path)

    store.save([Message(role="user", content="hello"), Message(role="assistant", content="hi")])

    assert store.load() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]


def test_store_clear(tmp_path):
    path = tmp_path / "conversation.json"
    store = ConversationStore(path)
    store.append_pair("one", "two")

    store.clear()

    assert store.load() == []
