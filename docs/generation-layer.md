# Grounded Gemini answer generation

The generation layer accepts a user question and ranked `RetrievalResult` values. It formats only the relevant grounding fields—file path, source lines, symbol, language, and chunk content—then sends a bounded prompt to Gemini.

The system instruction requires the model to use only retrieved repository context, avoid inventing code, identify inferences, and state when the context is insufficient. With no usable retrieved chunks, the service returns an explicit insufficiency response without calling Gemini.

`LLMProvider` keeps the application independent of a particular generation vendor. `GeminiProvider` is the current REST API adapter and reads `GEMINI_API_KEY` and `GEMINI_MODEL` through the application configuration; no model name is hardcoded in generation logic.

## Context limits

Chunks remain in retrieval rank order. At most `GENERATION_MAX_CONTEXT_CHUNKS` chunks are considered, and complete formatted chunks are added only while the combined context is within `GENERATION_MAX_CONTEXT_CHARS`. Oversized chunks are skipped rather than being partially copied into a prompt, preserving accurate source context.

This phase does not add citations to generated answers, hybrid retrieval, reranking, conversation memory, or a frontend chat experience.
