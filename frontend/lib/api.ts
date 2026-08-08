export type Repository = { repository_id: string; repository_url: string; repository_name: string; status: string; file_count: number; total_size_bytes: number; languages: string[]; indexed_chunk_count?: number; error?: string | null };
export type Source = { citation_id: string; file_path: string; start_line: number; end_line: number; symbol_name: string | null; chunk_id: string };
export type ChatResponse = { answer: string; sources: Source[]; conversation_id: string; retrieval: Array<{ standalone_query: string; rank: number; file_path: string; start_line: number; end_line: number; symbol_name: string | null; relevance_score: number }> };
export type IndexedChunk = { chunk_id: string; file_path: string; language: string; start_line: number; end_line: number; symbol_name: string | null; chunk_type: string; content: string };
export type RepositoryFile = { file_path: string; language: string; chunk_count: number };
export type SearchResult = IndexedChunk & { rank: number; relevance_score: number };

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : (response.json() as Promise<T>);
}

export const api = {
  indexRepository: (repositoryUrl: string) => request<Repository>("/repositories/index", { method: "POST", body: JSON.stringify({ repository_url: repositoryUrl }) }),
  getRepository: (repositoryId: string) => request<Repository>(`/repositories/${encodeURIComponent(repositoryId)}/status`),
  getFiles: (repositoryId: string) => request<{ repository_id: string; files: RepositoryFile[] }>(`/repositories/${encodeURIComponent(repositoryId)}/files`),
  getFile: (repositoryId: string, filePath: string) => request<IndexedChunk[]>(`/repositories/${encodeURIComponent(repositoryId)}/files/${filePath.split("/").map(encodeURIComponent).join("/")}`),
  chat: (payload: { repository_id: string; conversation_id?: string; message: string }) => request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(payload) }),
  search: (payload: { repository_id: string; query: string; top_k?: number }) => request<{ results: SearchResult[] }>("/repositories/search", { method: "POST", body: JSON.stringify(payload) }),
};
