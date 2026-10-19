import { useEffect, useRef, useState } from "react";
import Message from "./Message";

export default function Chat({ documents, messages, isLoading, onSend }) {
  const [input, setInput] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [includeTrace, setIncludeTrace] = useState(false);
  const scrollRef = useRef(null);
  const indexedDocs = documents.filter((doc) => doc.status === "indexed");

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function handleSubmit(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question || isLoading) return;
    onSend(question, documentId || null, includeTrace);
    setInput("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) handleSubmit(event);
  }

  return (
    <div className="chat">
      <div className="chat__toolbar">
        <label htmlFor="doc-filter" className="chat__filter-label">Ask about:</label>
        <select id="doc-filter" className="chat__filter" value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
          <option value="">All documents</option>
          {indexedDocs.map((doc) => <option key={doc.document_id} value={doc.document_id}>{doc.filename}</option>)}
        </select>
        <label className="pipeline-toggle">
          <input type="checkbox" checked={includeTrace} onChange={(event) => setIncludeTrace(event.target.checked)} />
          Show RAG pipeline
        </label>
      </div>

      <div className="chat__history">
        {messages.length === 0 && (
          <div className="chat__empty">
            <p className="main__title">Chat with your college notes</p>
            <p className="main__subtitle">Upload a PDF from the sidebar, wait for it to say “Ready”, then ask a question. Answers use your notes and include page citations.</p>
            <p className="main__subtitle">Enable “Show RAG pipeline” to inspect the actual queries, retrieved chunks, reranking scores, and context used for an answer.</p>
          </div>
        )}
        {messages.map((message, index) => <Message key={index} message={message} />)}
        {isLoading && (
          <div className="message message--assistant">
            <div className="message__avatar">🤖</div>
            <div className="message__body"><div className="message__bubble message__bubble--loading"><span className="dot" /><span className="dot" /><span className="dot" /></div></div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <form className="chat__input-row" onSubmit={handleSubmit}>
        <textarea
          className="chat__input"
          placeholder="Ask your notes... (Enter to send, Shift+Enter for a new line)"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button className="chat__send" type="submit" disabled={isLoading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}