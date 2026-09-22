/**
 * Citation list shown under an AI answer. `sources` comes directly from
 * the backend's response — it's built from real chunk metadata
 * (document filename + page number), never invented on the frontend.
 * Props:
 *   sources: [{ document, page, score }]
 */
export default function Sources({ sources }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources">
      <div className="sources__label">Sources</div>
      <ul className="sources__list">
        {sources.map((s) => (
          <li className="sources__item" key={`${s.document}-${s.page}`}>
            📄 {s.document} — Page {s.page}
          </li>
        ))}
      </ul>
    </div>
  );
}
