import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const FEATURES = [
  {
    icon: '∂',
    title: 'Gradient Norm Analysis',
    desc: 'Runs backpropagation on every generated token, measuring how unstable the model\'s output is mathematically.'
  },
  {
    icon: 'H',
    title: 'Softmax Entropy',
    desc: 'Measures how spread out the model\'s probability mass is across the entire vocabulary per token.'
  },
  {
    icon: 'σ',
    title: 'MC Dropout Variance',
    desc: 'Runs stochastic forward passes to test if the model changes its story when its neurons are randomly disabled.'
  },
  {
    icon: '⟳',
    title: 'Auto-RAG Healing',
    desc: 'When a hallucination is detected, the model halts and fetches verified context to heal the output.'
  }
];

// Animated ticker
function Ticker() {
  const phrases = [
    'MECHANISTIC INTERPRETABILITY',
    'GRADIENT ANALYSIS',
    'SOFTMAX ENTROPY',
    'MC DROPOUT VARIANCE',
    'REAL-TIME STREAMING',
    'QWEN 0.5B',
    'HALLUCINATION DETECTION',
  ];
  const text = phrases.join('  ·  ') + '  ·  ';
  return (
    <div className="ticker-wrap">
      <div className="ticker-track">
        <span>{text}{text}</span>
      </div>
    </div>
  );
}

// Animated blinking cursor
function Cursor() {
  return <span className="blink-cursor">|</span>;
}

// Neural Network 3D Animation Component
function NeuralAnimation() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let w, h;
    const NODES = 60;
    
    // Abstract 3D sphere representing an LLM
    const nodes = Array.from({ length: NODES }, () => {
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos((Math.random() * 2) - 1);
      return {
        theta, phi,
        speedTheta: (Math.random() - 0.5) * 0.02,
        speedPhi: (Math.random() - 0.5) * 0.02,
        r: 150
      };
    });

    function resize() {
      w = canvas.width = canvas.offsetWidth;
      h = canvas.height = canvas.offsetHeight;
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h / 2;

      // Update positions
      nodes.forEach(n => {
        n.theta += n.speedTheta;
        n.phi += n.speedPhi;
        
        // 3D to 2D projection
        const x3d = n.r * Math.sin(n.phi) * Math.cos(n.theta);
        const y3d = n.r * Math.cos(n.phi);
        const z3d = n.r * Math.sin(n.phi) * Math.sin(n.theta) + 300;
        
        const scale = 400 / z3d;
        n.px = cx + x3d * scale;
        n.py = cy + y3d * scale;
        n.scale = scale;
      });

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].px - nodes[j].px;
          const dy = nodes[i].py - nodes[j].py;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 80) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(255, 51, 51, ${0.3 * (1 - dist / 80)})`;
            ctx.lineWidth = 1;
            ctx.moveTo(nodes[i].px, nodes[i].py);
            ctx.lineTo(nodes[j].px, nodes[j].py);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      nodes.forEach((n, i) => {
        ctx.beginPath();
        ctx.arc(n.px, n.py, 2 * n.scale, 0, Math.PI * 2);
        ctx.fillStyle = i % 5 === 0 ? 'rgba(48, 209, 88, 0.8)' : `rgba(255, 255, 255, ${0.2 * n.scale})`;
        ctx.fill();
        
        if (i % 5 === 0) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = 'rgba(48, 209, 88, 0.8)';
            ctx.fill();
            ctx.shadowBlur = 0;
        }
      });

      animId = requestAnimationFrame(draw);
    }

    resize();
    draw();
    window.addEventListener('resize', resize);
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="llm-animation-container" />;
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [visible, setVisible] = useState(false);
  const [showIntro, setShowIntro] = useState(true);
  const [introFadeOut, setIntroFadeOut] = useState(false);

  useEffect(() => {
    // Start intro sequence
    const t1 = setTimeout(() => {
      setIntroFadeOut(true); // Start fading out intro
    }, 3000);

    const t2 = setTimeout(() => {
      setShowIntro(false); // Remove intro completely
      setVisible(true); // Fade in landing page
    }, 4500);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  return (
    <>
      {showIntro && (
        <div className={`intro-screen ${introFadeOut ? 'fade-out' : ''}`}>
          <div className="glass-text">HALLUCINATION</div>
        </div>
      )}

      <div className={`landing-page ${visible ? 'landing-visible' : ''}`}>

      {/* Top Nav */}
      <nav className="landing-nav">
        <span className="landing-logo">
          <span style={{ color: 'var(--accent-red)' }}>//</span> HM
        </span>
        <span className="landing-nav-tag">Capstone Project · 2026</span>
      </nav>

      {/* Hero: Split Layout */}
      <section className="split-hero">
        <div className="hero-left">
          <div className="landing-eyebrow">
            <span className="badge">LIVE DEMO READY</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Mechanistic Interpretability Engine
            </span>
          </div>

          <h1 className="title-massive">
            HALLUCI<br />
            <span style={{ color: 'var(--accent-red)' }}>NATION</span><br />
            <span style={{ WebkitTextStroke: '2px var(--text-primary)', color: 'transparent' }}>MONITOR</span>
          </h1>

          <p className="landing-subtitle">
            A real-time AI safety dashboard that hooks deep into a language model's
            internal math — catching hallucinations <em>as they are generated</em>,
            token by token. <Cursor />
          </p>

          <div>
            <button className="landing-cta" onClick={() => navigate('/dashboard')}>
              <span>GET STARTED</span>
              <span className="cta-arrow">→</span>
            </button>
          </div>
        </div>
        
        <div className="hero-right">
          <NeuralAnimation />
          {/* Annotation Lines simulating the reference image's layout */}
          <div className="annotation-line" style={{ top: '30%', left: '30%', width: '150px', transform: 'rotate(-25deg)' }} />
          <div className="annotation-text" style={{ top: '22%', left: '25%' }}>Gradient Norm</div>
          
          <div className="annotation-line" style={{ top: '65%', left: '40%', width: '120px', transform: 'rotate(15deg)' }} />
          <div className="annotation-text" style={{ top: '75%', left: '45%' }}>Dropout Variance</div>
          
          <div className="annotation-line" style={{ top: '45%', right: '20%', width: '180px', transform: 'rotate(170deg)', transformOrigin: 'right center' }} />
          <div className="annotation-text" style={{ top: '42%', right: '10%' }}>Softmax Entropy</div>
        </div>
      </section>

      {/* Ticker */}
      <Ticker />

      {/* Horizontal Carousel (Ref Image Layout) */}
      <section className="carousel-section">
        <p style={{ fontSize: '0.85rem', fontWeight: 900, color: 'var(--accent-red)', letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: '2rem' }}>HOW IT WORKS</p>
        <h2 className="carousel-header">Deep Model Inspection</h2>
        <div className="carousel-container">
          {FEATURES.map((f, i) => (
            <div key={i} className="carousel-card">
              <div className="carousel-icon">{f.icon}</div>
              <h3 className="carousel-title">{f.title}</h3>
              <p className="carousel-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Circular Stats Layout (Ref Image Layout) */}
      <section className="circular-stats-section">
        <div className="stats-left">
           <div className="dial-circle">
               {Array.from({ length: 40 }).map((_, i) => (
                   <div key={i} className="dial-tick" style={{ transform: `rotate(${i * 9}deg)` }} />
               ))}
               <div className="dial-center">
                   <div style={{ fontSize: '3.5rem', fontWeight: 900, lineHeight: 1 }}>500M</div>
                   <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: '0.5rem' }}>Parameters<br/>Monitored</div>
               </div>
           </div>
        </div>
        <div className="stats-right">
           <div className="stat-card">
              <div className="stat-card-val" style={{ color: 'var(--accent-red)' }}>3</div>
              <div className="stat-card-label">Uncertainty<br/>Signals Extracted</div>
           </div>
           <div className="stat-card">
              <div className="stat-card-val" style={{ color: 'var(--accent-green)' }}>~0ms</div>
              <div className="stat-card-label">Streaming<br/>Latency (SSE)</div>
           </div>
           <div className="stat-card">
              <div className="stat-card-val" style={{ color: '#fff' }}>600+</div>
              <div className="stat-card-label">Math operations<br/>Per Token</div>
           </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <span>Built by Ayaan Akhter · Capstone 2026</span>
        <span style={{ color: 'var(--accent-red)' }}>Hallucination Monitor</span>
      </footer>
    </div>
    </>
  );
}
