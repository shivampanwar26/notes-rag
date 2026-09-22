import { useEffect, useRef, useState } from "react";
import Message from "./Message";

/**
 * Props:
 *   documents: array of document records (for the "ask about" filter)
 *   messages: array of { role, text, sources?, isError? }
 *   isLoading: whether a request is in flight
 *   onSend(question, documentId): send a question, optionally scoped to one document
 */
export default function Chat({ documents, messages, isLoading, onSend }) {
  const [input, setInput] = useState("");
  const [documentId, setDocumentId] = useState(""); // "" = search all documents
  const scrollRef = useRef(null);

  const indexedDocs = documents.filter((d) => d.status === "indexed");

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function handleSubmit(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || isLoading) return;
    onSend(question, documentId || null);
    setInput("");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }

  return (
    <div className="chat">
      <div className="chat__toolbar">
        <label htmlFor="doc-filter" className="chat__filter-label">
          Ask about:
        </label>
        <select
          id="doc-filter"
          className="chat__filter"
          value={documentId}
          onChange={(e) => setDocumentId(e.target.value)}
        >
          <option value="">All documents</option>
          {indexedDocs.map((d) => (
            <option key={d.document_id} value={d.document_id}>
              {d.filename}
            </option>
          ))}
        </select>
      </div>

      <div className="chat__history">
        {messages.length === 0 && (
          <div className="chat__empty">
            <p className="main__title">Chat with your college notes</p>
            <p className="main__subtitle">
              Upload a PDF from the sidebar, wait for it to say "Ready", then
              ask a question below. Answers are grounded only in your
              uploaded notes, with page citations.
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} message={m} />
        ))}
        {isLoading && (
          <div className="message message--assistant">
            <div className="message__avatar">🤖</div>
            <div className="message__body">
              <div className="message__bubble message__bubble--loading">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <form className="chat__input-row" onSubmit={handleSubmit}>
        <textarea
          className="chat__input"
          placeholder="Ask your notes... (Enter to send, Shift+Enter for a new line)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button className="chat__send" type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
