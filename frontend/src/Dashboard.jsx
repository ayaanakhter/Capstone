import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import './index.css';

function LoadingState({ isFirstRun }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#ff3333' }}>
      <div className="spinner" style={{ marginBottom: '1rem', width: '40px', height: '40px', border: '3px solid #333', borderTop: '3px solid #ff3333', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
      <p style={{ fontFamily: 'Inter, sans-serif', textTransform: 'uppercase', letterSpacing: '2px', fontSize: '0.9rem', animation: 'pulse 1.5s infinite' }}>
        {isFirstRun ? "Waking up Qwen AI into Memory..." : "Analyzing prompt and extracting signals..."}
      </p>
    </div>
  );
}

function StatBlock({ label, value, max = 1, format = "fixed" }) {
  const isDanger = format === "percent" ? value > 0.5 : value > 0.6;
  const displayValue = format === "percent" 
    ? `${(value * 100).toFixed(0)}%` 
    : value.toFixed(3);
    
  const percentage = Math.min(Math.max(value / max, 0), 1) * 100;

  return (
    <div className="stat-block">
      <div className="stat-label">
        <span className="red-dot" style={{ width: 8, height: 8 }}></span>
        {label}
      </div>
      <div className={`stat-value ${isDanger ? 'danger' : ''}`}>
        {displayValue}
      </div>
      <div className="progress-bg">
        <div 
          className={`progress-fill ${isDanger ? 'danger' : ''}`} 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [tokens, setTokens] = useState([]);
  const [hasGeneratedBefore, setHasGeneratedBefore] = useState(false);
  
  const [currentScore, setCurrentScore] = useState(0);
  const [currentEntropy, setCurrentEntropy] = useState(0);
  const [currentGradNorm, setCurrentGradNorm] = useState(0);

  const endOfTokensRef = useRef(null);

  useEffect(() => {
    endOfTokensRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [tokens]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setTokens([]);
    setCurrentScore(0);
    setCurrentEntropy(0);
    setCurrentGradNorm(0);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: 60 })
      });

      setHasGeneratedBefore(true);


      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              if (data.event === "token") {
                setTokens(prev => [...prev, data]);
                setCurrentScore(data.overall_lie_score || data.lie_score);
                setCurrentEntropy(data.entropy);
                setCurrentGradNorm(data.gradient_norm);
              } else if (data.event === "end") {
                setIsGenerating(false);
              }
            } catch (e) {
              console.error("Error parsing JSON chunk:", e, dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error("Failed to generate:", error);
      setIsGenerating(false);
    }
  };

  return (
    <div className="app-container">
      {/* Left Main Content */}
      <div className="main-content">
        <h1 className="title-massive">HALLUCI<br/>NATION<br/>MONITOR</h1>
        
        <div className="output-box">
          {tokens.length === 0 && !isGenerating ? (
            <p style={{ color: 'var(--text-secondary)' }}>
              In a world full of black-box models, AI outputs are becoming harder to trust.
              <br/><br/>
              Enter a prompt below to monitor internal uncertainty and gradient signals in real time.
            </p>
          ) : tokens.length === 0 && isGenerating ? (
            <LoadingState isFirstRun={!hasGeneratedBefore} />
          ) : (
            <p>
              {tokens.map((t, i) => (
                <span 
                  key={i} 
                  className={`token ${t.is_flagged ? 'token-flagged' : ''}`}
                >
                  {t.token}
                </span>
              ))}
              {isGenerating && <span style={{ opacity: 0.5, animation: 'pulse 1s infinite' }}>...</span>}
              <span ref={endOfTokensRef} />
            </p>
          )}
        </div>

        <form className="input-container" onSubmit={handleGenerate}>
          <input 
            type="text" 
            placeholder="Type your prompt here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isGenerating}
          />
          <button type="submit" disabled={isGenerating || !prompt.trim()}>
            Run Analysis <ArrowRight size={24} />
          </button>
        </form>
      </div>

      {/* Right Sidebar */}
      <div className="sidebar">
        <h2>
          <span>SIGNALS</span>
          <span className="red-dot"></span>
        </h2>
        
        <div style={{ flexGrow: 1 }}>
          <StatBlock 
            value={currentScore} 
            label="Hallucination Risk" 
            max={1.0}
            format="percent"
          />
          
          <StatBlock 
            label="Entropy" 
            value={currentEntropy} 
            max={1.0} 
          />
          
          <StatBlock 
            label="Gradient Norm" 
            value={currentGradNorm} 
            max={1.0} 
          />
        </div>

        <div>
          <p style={{ fontSize: '0.8rem', color: '#555', textTransform: 'uppercase', fontWeight: 700 }}>
            Powered by Qwen-0.5B internal state analysis. Over 600 parameters checked per token.
          </p>
          <Link to="/architecture" className="bold-link" style={{ borderColor: '#111', color: '#111' }}>
            VIEW ARCHITECTURE
          </Link>
          <Link to="/history" className="bold-link" style={{ borderColor: '#ff3333', color: '#ff3333', marginTop: '0.75rem' }}>
            VIEW HISTORY
          </Link>
          <Link to="/compare" className="bold-link" style={{ borderColor: '#30d158', color: '#30d158', marginTop: '0.75rem' }}>
            ⚡ UGD COMPARISON
          </Link>
        </div>
      </div>
    </div>
  );
}
