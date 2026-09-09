import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Zap, ShieldAlert, Cpu, Globe, CheckCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import './index.css';

const API = 'http://localhost:8000';

function StandardToken({ token }) {
  if (token.is_flagged) {
    return (
      <span
        title={`Risk: ${Math.round(token.lie_score * 100)}% | Diagnosis: ${token.diagnosis}`}
        style={{
          background: 'rgba(255,51,51,0.18)',
          borderBottom: '2px solid #ff3333',
          color: '#ff9999',
          padding: '0 1px',
          cursor: 'help',
        }}
      >
        {token.token}
      </span>
    );
  }
  return <span style={{ color: '#e0e0e0' }}>{token.token}</span>;
}

export default function Dashboard() {
  const [prompt, setPrompt] = useState('');
  const [tokens, setTokens] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRagSearching, setIsRagSearching] = useState(false);
  const [ragResult, setRagResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [flaggedTokensList, setFlaggedTokensList] = useState([]);

  // 3D Tilt State
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!containerRef.current) return;
      const { innerWidth, innerHeight } = window;
      const x = (e.clientX / innerWidth - 0.5) * 10; // Max tilt 5deg
      const y = (e.clientY / innerHeight - 0.5) * -10;
      setTilt({ x: y, y: x });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setTokens([]);
    setStats(null);
    setRagResult(null);
    setFlaggedTokensList([]);
    setChartData([]);
    setIsGenerating(true);
    setIsRagSearching(false);

    try {
      const res = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 60 }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let allRisk = [], allEntropy = [], totalFlagged = 0;
      let localFlaggedList = [];
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
              setTokens(prev => [...prev, data]);
              allRisk.push(data.lie_score);
              allEntropy.push(data.entropy);
              
              if (data.is_flagged) {
                totalFlagged++;
                localFlaggedList.push(data);
              }

              const currentRisk = data.lie_score * 100;
              setChartData(prev => {
                const newData = [...prev];
                newData[tokenIndex] = { ...newData[tokenIndex], name: tokenIndex, Risk: currentRisk };
                return newData;
              });
              tokenIndex++;

            } else if (data.event === 'rag_search_start') {
              setIsRagSearching(true);
              if (allRisk.length) {
                const avg = arr => (arr.reduce((a, b) => a + b, 0) / arr.length);
                setStats({
                  'Hallucination Risk': `${Math.round(avg(allRisk) * 100)}%`,
                  'Avg Entropy':        `${Math.round(avg(allEntropy) * 100)}%`,
                  'Flagged Tokens':     `${totalFlagged}`,
                });
                setFlaggedTokensList(localFlaggedList);
              }
            } else if (data.event === 'rag_search_result') {
              setIsRagSearching(false);
              setRagResult(data.result);
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
      setIsRagSearching(false);
    }
  };

  return (
    <>
      <div className="deep-space-bg"><div className="stars"></div></div>
      
      {/* Premium Header - outside the 3D scene to keep it usable */}
      <nav style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '1.5rem 3rem', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0,
        background: 'rgba(5, 10, 16, 0.6)', backdropFilter: 'blur(20px)', position: 'sticky', top: 0, zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Link to="/" style={{ color: 'var(--text-primary)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
              <ArrowLeft size={18} />
            </Link>
            <div style={{ width: '2px', height: '24px', background: 'var(--border-color)' }} />
            <div>
              <div style={{ fontWeight: 900, fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#fff', lineHeight: 1.2 }}>
                HALLUCI<span style={{ color: 'var(--accent-red)' }}>NATION</span>
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 700 }}>
                Forensic Analysis Dashboard
              </div>
            </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <Link to="/history" style={{ color: 'var(--text-secondary)', textDecoration: 'none', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', transition: 'color 0.2s' }} onMouseEnter={e => e.target.style.color = '#fff'} onMouseLeave={e => e.target.style.color = 'var(--text-secondary)'}>
            View History
          </Link>
          <Link to="/architecture" className="landing-cta" style={{ padding: '0.6rem 1.2rem', fontSize: '0.8rem', borderRadius: '4px', textDecoration: 'none', boxShadow: '0 0 15px rgba(51,204,255,0.2)' }}>
            <Cpu size={14} /> 3D Architecture
          </Link>
        </div>
      </nav>

      <div className="dashboard-3d-wrapper" ref={containerRef} style={{ minHeight: 'calc(100vh - 80px)' }}>
        <div className="dashboard-3d-scene" style={{ 
          transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
          padding: '2rem 3rem', display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: '1400px', margin: '0 auto', width: '100%',
          position: 'relative'
        }}>
          
          {/* SVG Neural Sync Lines */}
          <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }}>
            {/* Input to Chart to Output */}
            <path d="M 900 150 C 700 150, 700 150, 600 150" fill="none" stroke="rgba(51,204,255,0.2)" strokeWidth="2" strokeDasharray="5,5" />
            <path d="M 900 350 C 700 350, 700 350, 600 350" fill="none" stroke="rgba(255,51,51,0.2)" strokeWidth="2" strokeDasharray="5,5" />
            {/* Output to Forensics and Ground Truth */}
            {(stats || isRagSearching) && (
              <>
                <path d="M 400 450 C 400 550, 300 550, 300 600" fill="none" stroke="rgba(255,51,51,0.4)" strokeWidth="2" />
                <circle cx="300" cy="600" r="4" fill="#ff3333" />
                <path d="M 500 450 C 500 550, 900 550, 900 600" fill="none" stroke="rgba(51,204,255,0.4)" strokeWidth="2" />
                <circle cx="900" cy="600" r="4" fill="#33ccff" />
              </>
            )}
          </svg>

          <div style={{ display: 'flex', gap: '3rem', position: 'relative', zIndex: 1 }}>
            {/* Main Output Panel */}
            <div className="glass-panel-dark" style={{
              flex: '1 1 60%', border: `1px solid rgba(51,204,255,0.3)`, display: 'flex', flexDirection: 'column', position: 'relative',
              boxShadow: '0 20px 40px rgba(0,0,0,0.5), inset 0 0 20px rgba(51,204,255,0.05)'
            }}>
              <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '2px', background: '#33ccff', boxShadow: `0 0 15px #33ccff` }} />
              <div style={{
                padding: '1.5rem', borderBottom: `1px solid rgba(51,204,255,0.1)`, background: `linear-gradient(180deg, rgba(51,204,255,0.05) 0%, transparent 100%)`,
                display: 'flex', alignItems: 'center', gap: '1rem'
              }}>
                <div style={{ width: 36, height: 36, borderRadius: '8px', background: `rgba(51,204,255,0.1)`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#33ccff' }}>
                  <Zap size={20} />
                </div>
                <div>
                  <div style={{ fontWeight: 900, fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Model Generation Output</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, marginTop: '0.25rem' }}>
                    Standard decoding — Live token tracking
                  </div>
                </div>
              </div>

              <div style={{ padding: '2rem', fontSize: '1.1rem', lineHeight: 2, minHeight: '300px' }}>
                {tokens.length === 0 && !isGenerating && (
                  <div style={{ color: '#555', fontStyle: 'italic', fontSize: '0.9rem', textAlign: 'center', marginTop: '4rem' }}>
                    Enter a prompt to initiate neural generation...
                  </div>
                )}
                <p style={{ margin: 0, fontFamily: 'Georgia, serif' }}>
                  {tokens.map((t, i) => <StandardToken key={i} token={t} />)}
                  {isGenerating && tokens.length > 0 && <span style={{ opacity: 0.4, animation: 'pulse 1s infinite' }}>▌</span>}
                </p>
              </div>
            </div>

            {/* Input & Chart Panel */}
            <div style={{ flex: '0 0 350px', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              {/* Input */}
              <div className="glass-panel-dark" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', border: '1px solid rgba(255,255,255,0.1)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                  Prompt Target
                </div>
                <textarea
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && !isGenerating && (e.preventDefault(), handleRun())}
                  placeholder="Ask a factual question..."
                  disabled={isGenerating}
                  style={{
                    width: '100%', height: '80px', background: 'transparent', border: 'none',
                    color: '#fff', fontSize: '1rem', fontFamily: 'Inter, sans-serif', outline: 'none', resize: 'none', padding: 0
                  }}
                />
                <button
                  className="landing-cta"
                  onClick={handleRun}
                  disabled={isGenerating || !prompt.trim()}
                  style={{ width: '100%', justifyContent: 'center', opacity: (isGenerating || !prompt.trim()) ? 0.5 : 1, padding: '1rem' }}
                >
                  <span>{isGenerating ? 'GENERATING...' : 'ANALYZE'}</span>
                </button>
              </div>

              {/* Chart */}
              <div className="glass-panel-dark" style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column', minHeight: '200px', border: '1px solid rgba(255,51,51,0.2)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1rem' }}>
                  Live Risk Chart
                </div>
                <div style={{ flex: 1, width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <YAxis stroke="#555" tick={{fontSize: 10}} domain={[0, 100]} tickLine={false} axisLine={false} hide />
                        <RechartsTooltip 
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #333', fontSize: '0.8rem', borderRadius: '4px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Line type="stepAfter" dataKey="Risk" stroke="#ff3333" strokeWidth={2} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Post-Generation Analysis section */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', transition: 'opacity 0.5s, transform 0.5s', opacity: (stats || isRagSearching) ? 1 : 0, transform: (stats || isRagSearching) ? 'translateY(0)' : 'translateY(20px)', position: 'relative', zIndex: 1 }}>
            
            {/* Hallucination Forensics */}
            <div className="glass-panel-dark" style={{ border: '1px solid rgba(255,51,51,0.3)', position: 'relative', boxShadow: '0 15px 30px rgba(255,51,51,0.1)' }}>
               <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '2px', background: '#ff3333', boxShadow: `0 0 15px #ff3333` }} />
               <div style={{ padding: '1.5rem', borderBottom: '1px solid rgba(255,51,51,0.1)', display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,51,51,0.05)' }}>
                  <ShieldAlert size={20} color="#ff3333" />
                  <div style={{ fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Hallucination Forensics</div>
               </div>
               <div style={{ padding: '1.5rem' }}>
                 {isGenerating && !stats ? (
                   <div style={{ color: '#555', fontStyle: 'italic', fontSize: '0.9rem' }}>Awaiting generation completion...</div>
                 ) : flaggedTokensList.length > 0 ? (
                   <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                     <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                       The Mechanistic Interpretability Engine detected high uncertainty in the following tokens:
                     </div>
                     {flaggedTokensList.map((t, idx) => (
                       <div key={idx} style={{ background: 'rgba(255,51,51,0.05)', borderLeft: '3px solid #ff3333', padding: '0.75rem', borderRadius: '0 4px 4px 0' }}>
                         <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                           <span style={{ fontWeight: 700, color: '#ff9999' }}>"{t.token}"</span>
                           <span style={{ fontSize: '0.75rem', color: '#ff3333', fontWeight: 900 }}>{Math.round(t.lie_score * 100)}% RISK</span>
                         </div>
                         <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                           <strong>Layer:</strong> Embedding Layer (Inferred)<br/>
                           <strong>Diagnosis:</strong> {t.diagnosis}
                         </div>
                       </div>
                     ))}
                   </div>
                 ) : (
                   <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#30d158' }}>
                     <CheckCircle size={18} />
                     <span>No significant hallucinations detected.</span>
                   </div>
                 )}
               </div>
            </div>

            {/* Ground Truth RAG */}
            <div className="glass-panel-dark" style={{ border: '1px solid rgba(51,204,255,0.3)', position: 'relative', boxShadow: '0 15px 30px rgba(51,204,255,0.1)' }}>
               <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '2px', background: '#33ccff', boxShadow: `0 0 15px #33ccff` }} />
               <div style={{ padding: '1.5rem', borderBottom: '1px solid rgba(51,204,255,0.1)', display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(51,204,255,0.05)' }}>
                  <Globe size={20} color="#33ccff" />
                  <div style={{ fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Ground Truth Context</div>
               </div>
               <div style={{ padding: '1.5rem', lineHeight: 1.6 }}>
                  {isRagSearching ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#33ccff', animation: 'pulse 1.5s infinite' }}>
                      <Globe size={16} /> Searching live internet for verified facts...
                    </div>
                  ) : ragResult ? (
                    <div style={{ fontSize: '0.95rem', color: '#e0e0e0' }}>
                      {ragResult}
                    </div>
                  ) : (
                    <div style={{ color: '#555', fontStyle: 'italic', fontSize: '0.9rem' }}>Awaiting search execution...</div>
                  )}
               </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}
