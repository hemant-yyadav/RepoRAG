# Conversational retrieval

`POST /chat` supports multi-turn repository questions using a process-local conversation ID. A conversation is partitioned by both repository ID and conversation ID, so history from one repository cannot influence another repository’s retrieval.

For a first turn, the user message is sent directly to retrieval. For follow-ups, the recent bounded history and latest message are sent through `QueryRewriter`, which uses the existing LLM provider abstraction to produce a standalone retrieval query. For example, “How does it validate the token?” can become “How does the authentication implementation validate JWT tokens?”

The retriever only receives the standalone query—not raw history. The pipeline then remains retrieval → reranking → bounded generation. If rewriting fails or returns no text, the original user message is used safely.

`CONVERSATION_MAX_HISTORY_MESSAGES` bounds retained user/assistant messages per conversation; `CONVERSATION_REWRITE_HISTORY_LENGTH` bounds the subset used in a rewrite prompt. Both prevent unlimited memory growth. Process-local history is intentionally ephemeral and resets when the backend restarts.

The response returns answer, verified sources, conversation ID, and retrieval diagnostics including the standalone query and final reranked chunks. No frontend chat UI is included.
