import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Trash2, RefreshCw } from 'lucide-react';

const API = 'http://localhost:8000';

function RiskBadge({ score }) {
  const pct = Math.round(score * 100);
  const color = score > 0.65 ? '#ff3333' : score > 0.35 ? '#ff9500' : '#30d158';
  const label = score > 0.65 ? 'HIGH' : score > 0.35 ? 'MED' : 'LOW';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
      background: `${color}22`, border: `1px solid ${color}`,
      color, fontSize: '0.72rem', fontWeight: 900,
      letterSpacing: '0.1em', padding: '0.2rem 0.6rem', borderRadius: '2px',
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block'
      }} />
      {label} · {pct}%
    </span>
  );
}

function MetricBar({ label, value, color = '#ff3333' }) {
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <span style={{ fontSize: '0.7rem', color: '#a0a0a0', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 700 }}>
          {label}
        </span>
        <span style={{ fontSize: '0.7rem', color: '#fff', fontWeight: 700 }}>
          {Math.round(value * 100)}%
        </span>
      </div>
      <div style={{ height: 4, background: '#2a2a2a', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.round(value * 100)}%`,
          background: color, borderRadius: 2,
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  );
}

function SessionCard({ session, onDelete }) {
  const [expanded, setExpanded] = useState(false);
  const date = session.created_at
    ? new Date(session.created_at).toLocaleString()
    : '—';

  return (
    <div className="history-card" style={{ animationDelay: '0s' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <RiskBadge score={session.hallucination_risk} />
            <span style={{ fontSize: '0.7rem', color: '#555', fontWeight: 600, letterSpacing: '0.05em' }}>
              {date}
            </span>
            <span style={{ fontSize: '0.7rem', color: '#555', fontWeight: 600 }}>
              {session.total_tokens} tokens · {session.flagged_tokens} flagged
            </span>
          </div>
          <p style={{ margin: 0, fontWeight: 700, fontSize: '1rem', color: '#fff', lineHeight: 1.4 }}>
            {session.prompt}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'transparent', border: '1px solid #333', color: '#a0a0a0',
              padding: '0.4rem 0.8rem', cursor: 'pointer', fontSize: '0.75rem',
              fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            {expanded ? 'COLLAPSE' : 'EXPAND'}
          </button>
          <button
            onClick={() => onDelete(session.id)}
            style={{
              background: 'transparent', border: '1px solid #333', color: '#ff3333',
              padding: '0.4rem 0.6rem', cursor: 'pointer', display: 'flex', alignItems: 'center',
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '1rem', marginTop: '1.25rem',
        padding: '1rem', background: '#161616', border: '1px solid #222',
      }}>
        <div>
          <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '0.4rem' }}>
            Hallucination Risk
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#ff3333', lineHeight: 1 }}>
            {Math.round(session.hallucination_risk * 100)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '0.4rem' }}>
            Entropy
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#fff', lineHeight: 1 }}>
            {Math.round(session.avg_entropy * 100)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '0.4rem' }}>
            Gradient Norm
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#fff', lineHeight: 1 }}>
            {Math.round(session.avg_gradient_norm * 100)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '0.4rem' }}>
            MC Variance
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#fff', lineHeight: 1 }}>
            {Math.round(session.avg_mc_variance * 100)}%
          </div>
        </div>
      </div>

      {/* Expanded: bars + full response */}
      {expanded && (
        <div style={{ marginTop: '1.25rem', padding: '1.25rem', background: '#161616', border: '1px solid #222' }}>
          <p style={{ fontSize: '0.7rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, margin: '0 0 1rem' }}>
            Signal Breakdown
          </p>
          <MetricBar label="Hallucination Risk" value={session.hallucination_risk} color="#ff3333" />
          <MetricBar label="Avg Entropy" value={session.avg_entropy} color="#ff9500" />
          <MetricBar label="Avg Gradient Norm" value={session.avg_gradient_norm} color="#0a84ff" />
          <MetricBar label="Avg MC Variance" value={session.avg_mc_variance} color="#30d158" />

          <p style={{ fontSize: '0.7rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, margin: '1.5rem 0 0.75rem' }}>
            AI Response
          </p>
          <p style={{ margin: 0, fontSize: '0.9rem', color: '#ccc', lineHeight: 1.8, fontStyle: 'italic', borderLeft: '2px solid #ff3333', paddingLeft: '1rem' }}>
            {session.response}
          </p>
        </div>
      )}
    </div>
  );
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/history?limit=50`);
      const data = await res.json();
      setSessions(data);
    } catch (e) {
      setError('Could not connect to the backend. Make sure the server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleDelete = async (id) => {
    await fetch(`${API}/history/${id}`, { method: 'DELETE' });
    setSessions(prev => prev.filter(s => s.id !== id));
  };

  const avgRisk = sessions.length
    ? (sessions.reduce((a, s) => a + s.hallucination_risk, 0) / sessions.length)
    : 0;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '2rem' }}>
      {/* Top Nav */}
      <nav style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #222', paddingBottom: '1rem', marginBottom: '3rem' }}>
        <Link to="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#a0a0a0', textDecoration: 'none', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          <ArrowLeft size={16} /> Back to Monitor
        </Link>
        <span style={{ fontSize: '0.75rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
          Session History
        </span>
      </nav>

      {/* Page Title */}
      <div style={{ marginBottom: '3rem' }}>
        <p style={{ fontSize: '0.7rem', color: '#ff3333', fontWeight: 900, letterSpacing: '0.2em', textTransform: 'uppercase', margin: '0 0 0.75rem' }}>
          DATABASE
        </p>
        <h1 style={{ fontSize: 'clamp(3rem, 7vw, 6rem)', fontWeight: 900, letterSpacing: '-0.04em', margin: '0 0 1rem', lineHeight: 0.9, textTransform: 'uppercase' }}>
          SESSION<br />
          <span style={{ color: '#ff3333' }}>HISTORY</span>
        </h1>
        <p style={{ color: '#a0a0a0', margin: 0 }}>
          Every prompt analyzed by the Hallucination Monitor is stored here — with full metrics.
        </p>
      </div>

      {/* Summary Stats */}
      {sessions.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', background: '#222', border: '1px solid #222', marginBottom: '2.5rem' }}>
          {[
            { label: 'Total Sessions', value: sessions.length },
            { label: 'Avg Hallucination Risk', value: `${Math.round(avgRisk * 100)}%` },
            { label: 'Total Tokens Analyzed', value: sessions.reduce((a, s) => a + s.total_tokens, 0).toLocaleString() },
          ].map((s, i) => (
            <div key={i} style={{ background: '#121212', padding: '1.5rem 2rem' }}>
              <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '0.5rem' }}>
                {s.label}
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 900, color: '#fff' }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <span style={{ fontSize: '0.8rem', color: '#555', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {sessions.length} session{sessions.length !== 1 ? 's' : ''} stored
        </span>
        <button
          onClick={fetchHistory}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            background: 'transparent', border: '1px solid #333',
            color: '#a0a0a0', padding: '0.5rem 1rem', cursor: 'pointer',
            fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.08em',
            textTransform: 'uppercase', fontFamily: 'Inter, sans-serif',
          }}
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Content */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '4rem', color: '#555' }}>
          <p style={{ textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, fontSize: '0.85rem' }}>
            Loading sessions...
          </p>
        </div>
      )}

      {error && (
        <div style={{ padding: '1.5rem', border: '1px solid #ff3333', color: '#ff3333', background: '#ff333311', marginBottom: '1.5rem' }}>
          {error}
        </div>
      )}

      {!loading && !error && sessions.length === 0 && (
        <div style={{ textAlign: 'center', padding: '6rem 2rem', border: '1px solid #222' }}>
          <p style={{ color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, fontSize: '0.85rem', margin: '0 0 1rem' }}>
            No sessions yet
          </p>
          <Link to="/dashboard" style={{ color: '#ff3333', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: '0.8rem' }}>
            Run your first analysis →
          </Link>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {sessions.map(s => (
          <SessionCard key={s.id} session={s} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  );
}
