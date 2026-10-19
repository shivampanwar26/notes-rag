import { useEffect, useRef, useState } from "react";
import api from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";

const POLL_INTERVAL_MS = 2500;

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const pollRef = useRef(null);

  async function loadDocuments() {
    const response = await api.get("/api/documents");
    setDocuments(response.data);
    return response.data;
  }

  useEffect(() => {
    loadDocuments().catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    const stillWorking = documents.some((doc) => ["uploaded", "processing"].includes(doc.status));
    clearInterval(pollRef.current);
    if (stillWorking) pollRef.current = setInterval(() => loadDocuments().catch(() => {}), POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [documents]);

  function handleUploaded(newDoc) {
    setDocuments((current) => [newDoc, ...current.filter((doc) => doc.document_id !== newDoc.document_id)]);
  }

  async function handleDelete(documentId) {
    await api.delete(`/api/documents/${documentId}`);
    setDocuments((current) => current.filter((doc) => doc.document_id !== documentId));
  }

  async function handleSend(question, documentId, includeTrace) {
    setMessages((current) => [...current, { role: "user", text: question }]);
    setIsChatLoading(true);
    try {
      const response = await api.post("/api/chat", {
        question,
        document_id: documentId,
        include_trace: includeTrace,
      });
      setMessages((current) => [...current, {
        role: "assistant",
        text: response.data.answer,
        sources: response.data.sources,
        trace: response.data.trace,
      }]);
    } catch (err) {
      const detail = err.response?.data?.detail || "Something went wrong reaching the server. Is the backend running?";
      setMessages((current) => [...current, { role: "assistant", text: detail, isError: true }]);
    } finally {
      setIsChatLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar documents={documents} onUploaded={handleUploaded} onDelete={handleDelete} />
      <main className="main">
        <Chat documents={documents} messages={messages} isLoading={isChatLoading} onSend={handleSend} />
      </main>
    </div>
  );
}
