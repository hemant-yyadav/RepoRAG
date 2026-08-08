# Repository-aware code intelligence

The code intelligence endpoints add practical file, symbol, and constrained natural-language inspection on top of the existing indexed chunks.

- `GET /repositories/{repository_id}/files/{file_path}` returns repository-isolated indexed chunks for one exact path.
- `GET /repositories/{repository_id}/symbols/{symbol_name}` returns chunks with the exact indexed symbol name.
- `GET /repositories/{repository_id}/search?query=...&file_path=...&symbol_name=...&language=...` performs natural-language search and applies any supplied metadata filters before vector retrieval and BM25 scoring.
- `POST /repositories/{repository_id}/files/explain` accepts `{ "file_path": "src/auth/jwt.py" }` and generates an explanation from only that file’s bounded indexed chunks.

All paths remain repository-scoped in Qdrant. The source content returned is the indexed chunk content, with exact file paths and line ranges. No graph database, call graph, or full-repository prompt is introduced.
