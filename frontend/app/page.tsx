export default function Home() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Phase 0 · Development scaffold</p>
        <h1>Codebase RAG Assistant</h1>
        <p className="summary">
          The project foundation is ready. Repository indexing, retrieval, and chat
          will arrive in later phases.
        </p>
      </section>
      <section className="status-card" aria-label="Backend status">
        <div>
          <p className="status-label">Backend health</p>
          <p className="status-value">Ready to check</p>
        </div>
        <code>{apiUrl}/health</code>
      </section>
    </main>
  );
}
