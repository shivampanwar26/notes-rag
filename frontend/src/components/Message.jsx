import Sources from "./Sources";
import PipelineExplorer from "./PipelineExplorer";

export default function Message({ message }) {
  const { role, text, sources, isError, trace } = message;
  return (
    <div className={`message message--${role}`}>
      <div className="message__avatar">{role === "user" ? "🧑" : "🤖"}</div>
      <div className="message__body">
        <div className={`message__bubble ${isError ? "message__bubble--error" : ""}`}>{text}</div>
        {role === "assistant" && !isError && <Sources sources={sources} />}
        {role === "assistant" && !isError && trace && <PipelineExplorer trace={trace} />}
      </div>
    </div>
  );
}
