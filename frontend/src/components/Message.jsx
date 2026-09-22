import Sources from "./Sources";

/**
 * One message in the conversation.
 * Props:
 *   message: { role: "user" | "assistant", text, sources?, isError? }
 */
export default function Message({ message }) {
  const { role, text, sources, isError } = message;

  return (
    <div className={`message message--${role}`}>
      <div className="message__avatar">{role === "user" ? "🧑" : "🤖"}</div>
      <div className="message__body">
        <div className={`message__bubble ${isError ? "message__bubble--error" : ""}`}>
          {text}
        </div>
        {role === "assistant" && !isError && <Sources sources={sources} />}
      </div>
    </div>
  );
}
