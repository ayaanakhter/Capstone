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
];

// Animated ticker — scrolls phrases across a horizontal marquee
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

// Animated number counter
function Counter({ target, suffix = '' }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = Math.ceil(target / 60);
    const interval = setInterval(() => {
      start += step;
      if (start >= target) { setVal(target); clearInterval(interval); }
      else setVal(start);
    }, 16);
    return () => clearInterval(interval);
  }, [target]);
  return <>{val.toLocaleString()}{suffix}</>;
}

export default function LandingPage() {
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [visible, setVisible] = useState(false);

  // Fade in on mount
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 50);
    return () => clearTimeout(t);
  }, []);

  // Particle canvas background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let w, h;
    const PARTICLES = 80;

    const particles = Array.from({ length: PARTICLES }, () => ({
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0003,
      vy: (Math.random() - 0.5) * 0.0003,
      r: Math.random() * 1.5 + 0.5,
      alpha: Math.random() * 0.5 + 0.1,
    }));

    function resize() {
      w = canvas.width = canvas.offsetWidth;
      h = canvas.height = canvas.offsetHeight;
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = (particles[i].x - particles[j].x) * w;
          const dy = (particles[i].y - particles[j].y) * h;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(255,51,51,${0.15 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i].x * w, particles[i].y * h);
            ctx.lineTo(particles[j].x * w, particles[j].y * h);
            ctx.stroke();
          }
        }
      }

      // Draw particles
      particles.forEach((p) => {
        p.x = (p.x + p.vx + 1) % 1;
        p.y = (p.y + p.vy + 1) % 1;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,51,51,${p.alpha})`;
        ctx.fill();
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

  return (
    <div className={`landing-page ${visible ? 'landing-visible' : ''}`}>

      {/* Particle Background */}
      <canvas ref={canvasRef} className="landing-canvas" />

      {/* Top Nav */}
      <nav className="landing-nav">
        <span className="landing-logo">
          <span style={{ color: 'var(--accent-red)' }}>//</span> HM
        </span>
        <span className="landing-nav-tag">Capstone Project · 2026</span>
      </nav>

      {/* Hero */}
      <section className="landing-hero">
        <div className="landing-eyebrow">
          <span className="badge">LIVE DEMO READY</span>
          <span className="badge-sep" />
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', letterSpacing: '0.1em' }}>
            MECHANISTIC INTERPRETABILITY ENGINE
          </span>
        </div>

        <h1 className="landing-title">
          HALLUCI<br />
          <span style={{ color: 'var(--accent-red)' }}>NATION</span><br />
          <span className="landing-title-outline">MONITOR</span>
        </h1>

        <p className="landing-subtitle">
          A real-time AI safety dashboard that hooks deep into a language model's<br />
          internal math — catching hallucinations <em>as they are generated</em>,<br />
          token by token. <Cursor />
        </p>

        <button
          className="landing-cta"
          onClick={() => navigate('/dashboard')}
        >
          <span>GET STARTED</span>
          <span className="cta-arrow">→</span>
        </button>
      </section>

      {/* Ticker */}
      <Ticker />

      {/* Stats Row */}
      <section className="landing-stats">
        <div className="stat-item">
          <span className="stat-number"><Counter target={600} suffix="+" /></span>
          <span className="stat-label">Parameters Monitored Per Token</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-number"><Counter target={3} /></span>
          <span className="stat-label">Uncertainty Signals Extracted</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-number"><Counter target={500} suffix="M" /></span>
          <span className="stat-label">Model Parameters (Qwen 0.5B)</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-number">~0ms</span>
          <span className="stat-label">Streaming Latency (SSE)</span>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="landing-features">
        <p className="section-label">HOW IT WORKS</p>
        <div className="feature-grid">
          {FEATURES.map((f, i) => (
            <div key={i} className="feature-card" style={{ animationDelay: `${i * 0.15}s` }}>
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="landing-bottom-cta">
        <h2 className="landing-bottom-title">
          Ready to see inside<br />
          <span style={{ color: 'var(--accent-red)' }}>the black box?</span>
        </h2>
        <button className="landing-cta landing-cta-outline" onClick={() => navigate('/dashboard')}>
          <span>LAUNCH MONITOR</span>
          <span className="cta-arrow">→</span>
        </button>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <span>Built by Ayaan Akhter · Capstone 2026</span>
        <span style={{ color: 'var(--accent-red)' }}>Hallucination Monitor</span>
      </footer>
    </div>
  );
}
