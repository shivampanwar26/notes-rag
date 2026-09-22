import { useRef, useState } from "react";
import api from "../api";

/**
 * Upload control shown in the sidebar.
 * Props:
 *   onUploaded: called with the new document record after a successful upload,
 *               so the parent (App) can refresh the document list.
 */
export default function FileUpload({ onUploaded }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(null); // { type: "error"|"success", message }

  async function uploadFile(file) {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus({ type: "error", message: "Only PDF files are accepted." });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    setStatus(null);
    try {
      const response = await api.post("/api/documents/upload", formData);
      setStatus({ type: "success", message: response.data.message });
      onUploaded(response.data.document);
    } catch (err) {
      const message =
        err.response?.data?.detail || "Upload failed. Please try again.";
      setStatus({ type: "error", message });
    } finally {
      setIsUploading(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    uploadFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div
      className={`uploader ${isDragging ? "uploader--active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <span className="uploader__label">Drag a PDF here, or</span>
      <button
        className="uploader__button"
        disabled={isUploading}
        onClick={() => inputRef.current?.click()}
      >
        {isUploading ? "Uploading..." : "+ Upload PDF"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        hidden
        onChange={(e) => uploadFile(e.target.files?.[0])}
      />
      {status && (
        <div className={`uploader__status uploader__status--${status.type}`}>
          {status.message}
        </div>
      )}
    </div>
  );
}
