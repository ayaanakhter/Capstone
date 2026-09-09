import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Zap, Shield, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import './index.css';

const API = 'http://localhost:8000';

// Token rendering for standard mode
function StandardToken({ token }) {
  if (token.is_flagged) {
    return (
      <span
        title={`Risk: ${Math.round(token.lie_score * 100)}%`}
        style={{
          background: 'rgba(255,51,51,0.18)',
          borderBottom: '2px solid #ff3333',
          color: '#ff9999',
          padding: '0 1px',
          cursor: 'default',
        }}
      >
        {token.token}
      </span>
    );
  }
  return <span style={{ color: '#e0e0e0' }}>{token.token}</span>;
}

// Token rendering for UGD mode
function UGDToken({ token }) {
  if (token.status === 'accepted') {
    return <span style={{ color: '#e0e0e0' }}>{token.token}</span>;
  }

  if (token.status === 'warned') {
    return (
      <span
        title={`Risky token: ${token.diagnosis || 'High Hallucination Risk'} (${Math.round(token.risk_score * 100)}%)`}
        style={{
          background: 'rgba(255,149,0,0.15)',
          borderBottom: '2px solid #ff9500',
          color: '#ffb830',
          padding: '0 1px',
          cursor: 'help',
        }}
      >
        {token.token}
      </span>
    );
  }

  if (token.status === 'retracted') {
    return (
      <span 
        title={`Retracted due to: ${token.diagnosis || 'High Hallucination Risk'} (${Math.round(token.risk_score * 100)}%)`}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
          background: 'rgba(255,51,51,0.12)', border: '1px solid #ff3333',
          color: '#ff3333', padding: '0.2rem 0.75rem', borderRadius: '2px',
          fontSize: '0.8rem', fontWeight: 900, letterSpacing: '0.05em',
          marginLeft: '4px', cursor: 'help'
        }}
      >
        🛑 STOPPED — model was about to say "<em style={{ fontStyle: 'italic', fontWeight: 400 }}>{token.original_token}</em>"
      </span>
    );
  }

  if (token.status === 'grounded') {
    return (
      <span
        title="Self-Healed: Token generated using verified Wikipedia context"
        style={{
          color: '#33ccff',
          textShadow: '0 0 8px rgba(51, 204, 255, 0.4)',
        }}
      >
        {token.token}
      </span>
    );
  }

  return <span>{token.token}</span>;
}

function Panel({ title, icon, accentColor, tokens, isGenerating, isRagSearching, mode, stats, retracted, correctedCount, loadingMsg }) {
  const isUGD = mode === 'ugd';
  return (
    <div style={{
      flex: 1,
      border: `1px solid ${accentColor}44`,
      background: '#111',
      display: 'flex',
      flexDirection: 'column',
      minHeight: 0,
    }}>
      {/* Panel Header */}
      <div style={{
        padding: '1rem 1.5rem',
        borderBottom: `1px solid ${accentColor}44`,
        background: `${accentColor}08`,
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        flexShrink: 0,
      }}>
        <span style={{ color: accentColor, display: 'flex' }}>{icon}</span>
        <div>
          <div style={{ fontWeight: 900, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#fff' }}>
            {title}
          </div>
          <div style={{ fontSize: '0.65rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
            {isUGD ? 'Token gating active — accept / correct / retract' : 'Standard greedy decoding — no safety gating'}
          </div>
        </div>
        {retracted && (
          <span style={{ marginLeft: 'auto', fontSize: '0.7rem', fontWeight: 900, color: '#ff3333', letterSpacing: '0.1em' }}>
            STOPPED EARLY
          </span>
        )}
      </div>

      {/* Output */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '1.5rem',
        fontSize: '1rem', lineHeight: 1.9, minHeight: 180,
      }}>
        {tokens.length === 0 && !isGenerating && !isRagSearching && (
          <p style={{ color: '#444', margin: 0, fontStyle: 'italic', fontSize: '0.9rem' }}>
            Output will appear here once you run the analysis...
          </p>
        )}
        {tokens.length === 0 && isGenerating && !isRagSearching && (
          <p style={{ color: '#555', margin: 0, fontSize: '0.85rem', animation: 'pulse 1.2s infinite', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
            {loadingMsg}
          </p>
        )}
        <p style={{ margin: 0 }}>
          {tokens.map((t, i) =>
            isUGD
              ? <UGDToken key={i} token={t} />
              : <StandardToken key={i} token={t} />
          )}
          {isGenerating && tokens.length > 0 && !isRagSearching && <span style={{ opacity: 0.4, animation: 'pulse 1s infinite' }}>▌</span>}
        </p>
        
        {/* RAG Searching UI */}
        {isRagSearching && (
          <div style={{
            marginTop: '1rem', padding: '0.75rem', background: 'rgba(51, 204, 255, 0.1)',
            borderLeft: '2px solid #33ccff', color: '#33ccff', fontSize: '0.85rem',
            animation: 'pulse 1.2s infinite', display: 'flex', alignItems: 'center', gap: '0.5rem'
          }}>
            🔍 <b>Self-Healing:</b> Searching Wikipedia for ground truth...
          </div>
        )}
      </div>

      {/* Stats Footer */}
      {stats && (
        <div style={{
          borderTop: `1px solid ${accentColor}33`,
          padding: '0.75rem 1.5rem',
          display: 'flex', gap: '1.5rem', flexShrink: 0,
          background: '#0d0d0d',
        }}>
          {Object.entries(stats).map(([k, v]) => (
            <div key={k}>
              <div style={{ fontSize: '0.6rem', color: '#444', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>{k}</div>
              <div style={{ fontSize: '1rem', fontWeight: 900, color: accentColor }}>{v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const [prompt, setPrompt] = useState('');

  const [stdTokens, setStdTokens]       = useState([]);
  const [ugdTokens, setUgdTokens]       = useState([]);
  const [stdGenerating, setStdGenerating] = useState(false);
  const [ugdGenerating, setUgdGenerating] = useState(false);
  const [stdStats, setStdStats]         = useState(null);
  const [ugdStats, setUgdStats]         = useState(null);
  const [ugdRetracted, setUgdRetracted] = useState(false);
  const [isRagSearching, setIsRagSearching] = useState(false);
  const [correctedCount, setCorrectedCount] = useState(0);

  // Live Chart Data
  const [chartData, setChartData] = useState([]);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setStdTokens([]);
    setUgdTokens([]);
    setStdStats(null);
    setUgdStats(null);
    setUgdRetracted(false);
    setIsRagSearching(false);
    setCorrectedCount(0);
    setChartData([]);
    setStdGenerating(true);
    setUgdGenerating(true);

    // Run both streams in parallel
    const runStandard = async () => {
      const res = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 40 }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let allRisk = [], allEntropy = [], allGrad = [], totalFlagged = 0;
      let tokenIndex = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.event === 'token') {
              setStdTokens(prev => [...prev, data]);
              allRisk.push(data.lie_score);
              allEntropy.push(data.entropy);
              allGrad.push(data.gradient_norm);
              if (data.is_flagged) totalFlagged++;

              const currentRisk = data.lie_score * 100;
              setChartData(prev => {
                const newData = [...prev];
                if (!newData[tokenIndex]) newData[tokenIndex] = { name: tokenIndex };
                newData[tokenIndex].stdRisk = currentRisk;
                return newData;
              });
              tokenIndex++;
            }
          } catch (_) {}
        }
      }
      setStdGenerating(false);
      if (allRisk.length) {
        const avg = arr => (arr.reduce((a, b) => a + b, 0) / arr.length);
        setStdStats({
          'Hallucination Risk': `${Math.round(avg(allRisk) * 100)}%`,
          'Avg Entropy':        `${Math.round(avg(allEntropy) * 100)}%`,
          'Flagged Tokens':     `${totalFlagged}`,
        });
      }
    };

    const runUGD = async () => {
      const res = await fetch(`${API}/generate/ugd`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 40 }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let allRisk = [], allEntropy = [], allGrad = [], corrected = 0, wasRetracted = false;
      let tokenIndex = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.event === 'token') {
              setUgdTokens(prev => [...prev, data]);
              allRisk.push(data.risk_score);
              allEntropy.push(data.entropy);
              allGrad.push(data.gradient_norm);
              if (data.status === 'corrected') corrected++;
              if (data.status === 'retracted') {
                wasRetracted = true;
                setUgdRetracted(true);
              }
              
              const currentRisk = data.risk_score * 100;
              setChartData(prev => {
                const newData = [...prev];
                if (!newData[tokenIndex]) newData[tokenIndex] = { name: tokenIndex };
                newData[tokenIndex].ugdRisk = currentRisk;
                return newData;
              });
              tokenIndex++;

            } else if (data.event === 'rag_search_start') {
              setIsRagSearching(true);
            } else if (data.event === 'rag_search_result') {
              setIsRagSearching(false);
            } else if (data.event === 'end') {
              setCorrectedCount(data.corrected_count || corrected);
            }
          } catch (_) {}
        }
      }
      setUgdGenerating(false);
      if (allRisk.length) {
        const avg = arr => (arr.reduce((a, b) => a + b, 0) / arr.length);
        setUgdStats({
          'Hallucination Risk': `${Math.round(avg(allRisk) * 100)}%`,
          'Avg Entropy':        `${Math.round(avg(allEntropy) * 100)}%`,
          'Tokens Corrected':   `${corrected}`,
          'Retracted':          wasRetracted ? 'YES' : 'NO',
        });
      }
    };

    // Fire both simultaneously
    runStandard();
    runUGD();
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0a0a0a' }}>
      {/* Top Nav */}
      <nav style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '1rem 2rem', borderBottom: '1px solid #1e1e1e', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="#fff" />
            <span style={{ fontWeight: 900, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#fff' }}>HALLUCI<br/>NATION</span>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontWeight: 900, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#fff' }}>UGD DASHBOARD</div>
          <div style={{ fontSize: '0.6rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
            Live Evaluation & Charting
          </div>
        </div>
        <Link to="/history" style={{ color: '#555', textDecoration: 'none', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          History →
        </Link>
      </nav>

      {/* Legend */}
      <div style={{ padding: '0.6rem 2rem', borderBottom: '1px solid #1a1a1a', display: 'flex', gap: '2rem', flexShrink: 0, background: '#0d0d0d' }}>
        <span style={{ fontSize: '0.7rem', color: '#555', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>UGD Token Legend:</span>
        <span style={{ fontSize: '0.7rem', color: '#e0e0e0', fontWeight: 600 }}>⬜ Accepted (safe)</span>
        <span style={{ fontSize: '0.7rem', color: '#ffb830', fontWeight: 600 }}>🟠 Warned (risky — hover for score)</span>
        <span style={{ fontSize: '0.7rem', color: '#ff3333', fontWeight: 600 }}>🛑 Retracted (generation stopped)</span>
        <span style={{ fontSize: '0.7rem', color: '#33ccff', fontWeight: 600 }}>🔵 Self-Healed (Auto-RAG)</span>
        <span style={{ fontSize: '0.7rem', color: '#ff9999', fontWeight: 600, marginLeft: 'auto' }}>Standard: <span style={{ borderBottom: '2px solid #ff3333' }}>underline = flagged</span></span>
      </div>

      {/* Panels Area */}
      <div style={{ flex: '1 1 50%', display: 'flex', gap: '1px', padding: '0', background: '#1a1a1a', overflow: 'hidden' }}>
        <Panel
          title="Standard Decoding"
          icon={<Zap size={18} />}
          accentColor="#ff3333"
          tokens={stdTokens}
          isGenerating={stdGenerating}
          mode="standard"
          stats={stdStats}
          retracted={false}
          correctedCount={0}
          loadingMsg="Generating without safety gating..."
        />
        <Panel
          title="UGD — Uncertainty-Gated Decoding"
          icon={<Shield size={18} />}
          accentColor="#30d158"
          tokens={ugdTokens}
          isGenerating={ugdGenerating}
          isRagSearching={isRagSearching}
          mode="ugd"
          stats={ugdStats}
          retracted={ugdRetracted}
          correctedCount={correctedCount}
          loadingMsg="UGD active — analyzing each token before emitting..."
        />
      </div>

      {/* Live Chart Area */}
      <div style={{ flex: '1 1 35%', minHeight: '200px', padding: '1rem 2rem', background: '#0a0a0a', borderTop: '1px solid #1a1a1a', display: 'flex', flexDirection: 'column' }}>
        <div style={{ fontSize: '0.7rem', color: '#555', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>
          Live Risk Tracking (Token by Token)
        </div>
        <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="name" stroke="#555" tick={{fontSize: 10}} />
                <YAxis stroke="#555" tick={{fontSize: 10}} domain={[0, 100]} />
                <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '0.8rem' }}
                    itemStyle={{ color: '#fff' }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '0.8rem' }} />
                <Line type="monotone" dataKey="stdRisk" name="Standard Risk %" stroke="#ff3333" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="ugdRisk" name="UGD Risk %" stroke="#30d158" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
        </div>
      </div>

      {/* Input Bar */}
      <div style={{
        padding: '1rem 2rem', borderTop: '1px solid #1a1a1a',
        display: 'flex', gap: '1rem', flexShrink: 0, background: '#0d0d0d',
      }}>
        <input
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !stdGenerating && !ugdGenerating && handleRun()}
          placeholder='Try a hallucination-prone prompt e.g. "When was World War 3 fought?" or "Prove 4 > 5"'
          disabled={stdGenerating || ugdGenerating}
          style={{
            flex: 1, background: '#111', border: '1px solid #2a2a2a',
            color: '#fff', padding: '0.8rem 1rem', fontSize: '0.9rem',
            fontFamily: 'Inter, sans-serif', outline: 'none',
          }}
        />
        <button
          onClick={handleRun}
          disabled={stdGenerating || ugdGenerating || !prompt.trim()}
          style={{
            background: (stdGenerating || ugdGenerating) ? '#1a1a1a' : '#30d158',
            color: (stdGenerating || ugdGenerating) ? '#555' : '#000',
            border: 'none', padding: '0.8rem 2rem',
            fontFamily: 'Inter, sans-serif', fontWeight: 900, fontSize: '0.85rem',
            textTransform: 'uppercase', letterSpacing: '0.1em',
            cursor: (stdGenerating || ugdGenerating) ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {stdGenerating || ugdGenerating ? 'RUNNING...' : 'RUN BOTH'}
        </button>
      </div>
    </div>
  );
}
