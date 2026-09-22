import { useEffect, useRef, useState } from "react";
import api from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";

// While any document is still "uploaded" or "processing", poll the
// document list so status badges (and the chat filter dropdown) update
// automatically once indexing finishes -- no manual refresh needed.
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
    loadDocuments();
    return () => clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    const stillWorking = documents.some(
      (d) => d.status === "uploaded" || d.status === "processing"
    );
    clearInterval(pollRef.current);
    if (stillWorking) {
      pollRef.current = setInterval(loadDocuments, POLL_INTERVAL_MS);
    }
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]);

  function handleUploaded(newDoc) {
    setDocuments((prev) => [newDoc, ...prev]);
  }

  async function handleDelete(documentId) {
    await api.delete(`/api/documents/${documentId}`);
    setDocuments((prev) => prev.filter((d) => d.document_id !== documentId));
  }

  async function handleSend(question, documentId) {
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setIsChatLoading(true);
    try {
      const response = await api.post("/api/chat", {
        question,
        document_id: documentId,
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: response.data.answer,
          sources: response.data.sources,
        },
      ]);
    } catch (err) {
      const detail =
        err.response?.data?.detail ||
        "Something went wrong reaching the server. Is the backend running?";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: detail, isError: true },
      ]);
    } finally {
      setIsChatLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        documents={documents}
        onUploaded={handleUploaded}
        onDelete={handleDelete}
      />
      <main className="main">
        <Chat
          documents={documents}
          messages={messages}
          isLoading={isChatLoading}
          onSend={handleSend}
        />
      </main>
    </div>
  );
}
