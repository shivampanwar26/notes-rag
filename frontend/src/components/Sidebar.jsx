import FileUpload from "./FileUpload";

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const STATUS_LABEL = {
  uploaded: "Queued",
  processing: "Indexing…",
  indexed: "Ready",
  failed: "Failed",
};

/**
 * Left-hand panel: app identity, list of uploaded documents (with live
 * indexing status), upload control.
 * Props:
 *   documents: array of document records from the backend
 *   onUploaded(doc): called when a new file finishes uploading
 *   onDelete(documentId): called when the user clicks the delete icon
 */
export default function Sidebar({ documents, onUploaded, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">College Notes AI</div>

      <div className="sidebar__section-label">
        DOCUMENTS ({documents.length})
      </div>
      <ul className="doc-list">
        {documents.length === 0 && (
          <li className="doc-list__empty">No documents yet. Upload one below.</li>
        )}
        {documents.map((doc) => (
          <li className="doc-item" key={doc.document_id}>
            <div className="doc-item__main">
              <span className="doc-item__name">📄 {doc.filename}</span>
              <span className="doc-item__meta">
                {formatSize(doc.size_bytes)}
                {doc.status === "indexed" &&
                  ` · ${doc.page_count} pages · ${doc.chunk_count} chunks`}
                {" · "}
                <span className={`status-badge status-badge--${doc.status}`}>
                  {STATUS_LABEL[doc.status] || doc.status}
                </span>
              </span>
              {doc.status === "failed" && doc.error && (
                <span className="doc-item__error" title={doc.error}>
                  {doc.error}
                </span>
              )}
            </div>
            <button
              className="doc-item__delete"
              title="Delete"
              onClick={() => onDelete(doc.document_id)}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>

      <FileUpload onUploaded={onUploaded} />
    </aside>
  );
}
