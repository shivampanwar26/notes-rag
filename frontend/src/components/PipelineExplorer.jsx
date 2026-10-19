import { useState } from "react";
import "./PipelineExplorer.css";

function Score({ value }) {
  if (value === null || value === undefined) return <span>—</span>;
  return <span>{Number(value).toFixed(4)}</span>;
}

function ChunkCard({ chunk, index, scoreLabel, scoreValue }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <article className="pipeline-chunk">
      <button className="pipeline-chunk__heading" onClick={() => setExpanded((value) => !value)}>
        <span><b>#{index + 1}</b> {chunk.source} · page {chunk.page}</span>
        <span>
          {scoreLabel && <>{scoreLabel}: <Score value={scoreValue} /> </>}
          {expanded ? "−" : "+"}
        </span>
      </button>
      {expanded && <>
        <div className="pipeline-chunk__id">{chunk.chunk_id}</div>
        <pre className="pipeline-chunk__text">{chunk.text}</pre>
      </>}
    </article>
  );
}

function Stage({ title, summary, children, defaultOpen = false }) {
  return (
    <details className="pipeline-stage" open={defaultOpen || undefined}>
      <summary><span>{title}</span><small>{summary}</small></summary>
      <div className="pipeline-stage__content">{children}</div>
    </details>
  );
}

export default function PipelineExplorer({ trace }) {
  const [open, setOpen] = useState(false);
  const queries = trace.queries || [];
  const searches = trace.search_results || [];
  const before = trace.before_reranking || [];
  const after = trace.after_reranking || [];

  return (
    <section className="pipeline-explorer">
      <button className="pipeline-explorer__toggle" onClick={() => setOpen((value) => !value)}>
        {open ? "Hide RAG pipeline" : "View RAG pipeline"}
        <span>{trace.retrieval_time_ms ?? "—"} ms retrieval</span>
      </button>
      {open && <div className="pipeline-explorer__body">
        <Stage title="1. Original query and query expansion" summary={`${queries.length} queries`} defaultOpen>
          <ol className="pipeline-query-list">{queries.map((query, index) => <li key={`${index}-${query}`}><b>{index === 0 ? "Original" : `Alternative ${index}`}</b><p>{query}</p></li>)}</ol>
          {trace.expansion_error && <p className="pipeline-warning">Query expansion issue: {trace.expansion_error}. Retrieval continued with available queries.</p>}
        </Stage>

        <Stage title="2. Hybrid retrieval (vector + BM25, fused with RRF)" summary={`${searches.length} queries`}>
          <p className="pipeline-explainer">Each query runs vector and BM25 search; their rankings are merged with Reciprocal Rank Fusion.</p>
          {searches.map((search, index) => <div className="pipeline-search" key={`${search.query_label}-${index}`}>
            <div className="pipeline-search__title"><b>{search.query_label}</b><span>{search.results.length} chunks</span></div>
            <p className="pipeline-search__query">Query: {search.query}</p>
            {search.results.length === 0
              ? <p className="pipeline-muted">No matches returned.</p>
              : search.results.map((chunk, chunkIndex) => <ChunkCard key={chunk.chunk_id} chunk={chunk} index={chunkIndex} />)}
          </div>)}
        </Stage>

        <Stage title="3. Merged candidates (before reranking)" summary={`${before.length} unique chunks`}>
          {before.map((chunk, index) => <ChunkCard key={chunk.chunk_id} chunk={chunk} index={index} />)}
        </Stage>

        <Stage title="4. After cross-encoder reranking" summary={`${after.length} final chunks`} defaultOpen>
          <p className="pipeline-explainer">The cross-encoder scores each candidate against the original question and keeps the top results.</p>
          {after.map((chunk, index) => <ChunkCard key={chunk.chunk_id} chunk={chunk} index={index} scoreLabel="Cross-encoder" scoreValue={chunk.rerank_score} />)}
        </Stage>

        <Stage title="5. Exact context sent to Gemini" summary={`${after.length} chunks`}>
          <pre className="pipeline-context">{trace.final_context || "No context was sent because retrieval returned no chunks."}</pre>
        </Stage>

        <Stage title="6. Pipeline settings and timing" summary={`${trace.retrieval_time_ms ?? "—"} ms retrieval`}>
          <pre className="pipeline-context">{JSON.stringify(trace.settings || {}, null, 2)}</pre>
          <p className="pipeline-muted">Retrieval time excludes Gemini answer generation and frontend rendering.</p>
        </Stage>
      </div>}
    </section>
  );
}