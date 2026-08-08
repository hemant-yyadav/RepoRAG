# Grounded citations and source tracking

The backend assigns stable citation IDs to the bounded retrieved chunks in rank order. Gemini receives context in this form:

```text
[1]
File: src/auth/jwt.py
Lines: 31-58
Symbol: validateToken
Language: Python
content:
...
```

The prompt instructs Gemini to place these IDs beside factual repository claims. The model never supplies source metadata: the backend keeps the file path, line range, symbol, and chunk ID in `SourceCitation` objects created directly from retrieved chunks.

After generation, citation IDs are parsed from the answer and validated against the IDs actually supplied in the prompt. Unknown IDs are removed from the displayed answer and never returned as sources. Repeated valid IDs produce one source entry; sources are ordered by their first appearance in the answer.

`POST /answers/generate` performs retrieval and grounded generation, returning an answer and verified `sources`. It adds no hybrid retrieval, reranking, conversation memory, or frontend behavior.
