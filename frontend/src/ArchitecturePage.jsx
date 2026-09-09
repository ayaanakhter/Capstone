import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import Architecture3D from './components/Architecture3D';
import './index.css';

const LAYER_INFO = {
  0: {
    title: "INPUT EMBEDDINGS",
    color: "#3366ff",
    description: "The very first layer of the model. When you type a prompt, it's converted into numbers (tokens) and mapped to high-dimensional mathematical vectors here.",
    signal: "Gradient Norm",
    signalDesc: "We backpropagate the final output probability all the way down to these input vectors. If the gradient is extremely large (a massive spike), it means the model is highly sensitive to the exact wording of your prompt—a classic sign of a hallucination or an adversarial attack."
  },
  1: {
    title: "ATTENTION & MLP BLOCKS",
    color: "#aa00ff",
    description: "The hidden 'brain' of the model. Self-Attention heads decide which words in the prompt relate to each other, and Multi-Layer Perceptrons (MLPs) act as the memory banks retrieving stored facts.",
    signal: "MC Dropout Variance",
    signalDesc: "We temporarily activate 'Dropout' to randomly disable some of these neurons, and run the model multiple times. If the model is confident and truly knows the answer, the output stays the same. If it's guessing, disabling even a few neurons causes the output to wildly change (High Variance)."
  },
  2: {
    title: "LM HEAD & SOFTMAX",
    color: "#ff3333",
    description: "The final layer. After all the processing is done, this layer outputs a probability distribution over the entire vocabulary (all ~50,000 possible next words).",
    signal: "Softmax Entropy",
    signalDesc: "We calculate the mathematical entropy of this probability distribution. Low entropy means the model is 99% sure the next word is 'Paris'. High entropy means the model's confidence is completely spread out across dozens of random words, indicating deep uncertainty."
  }
};

export default function ArchitecturePage() {
  const [activeLayer, setActiveLayer] = useState(null);

  const activeInfo = activeLayer !== null ? LAYER_INFO[activeLayer] : null;

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', background: '#0a0a0a', overflow: 'hidden' }}>
      
      {/* 3D Canvas Background */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
        <Architecture3D activeLayer={activeLayer} onSelectLayer={setActiveLayer} />
      </div>

      {/* UI Overlay */}
      <div style={{ position: 'absolute', top: 0, left: 0, padding: '2rem', zIndex: 10, width: '35%', minWidth: '400px', pointerEvents: 'none' }}>
        
        <Link to="/" className="bold-link" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem', pointerEvents: 'auto' }}>
          <ArrowLeft size={20} /> BACK TO DASHBOARD
        </Link>
        
        <h1 className="title-medium" style={{ color: 'white', marginBottom: '1rem', pointerEvents: 'auto' }}>
          NEURAL<br/>ARCHITECTURE
        </h1>
        
        {/* Dynamic Information Panel */}
        <div style={{ 
          background: 'rgba(10, 10, 10, 0.85)', 
          padding: '2rem', 
          border: `1px solid ${activeInfo ? activeInfo.color : '#333'}`,
          backdropFilter: 'blur(10px)',
          transition: 'all 0.3s ease',
          pointerEvents: 'auto'
        }}>
          {activeInfo ? (
            <div className="fade-in">
              <h3 style={{ color: activeInfo.color, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className="red-dot" style={{ background: activeInfo.color }}></span>
                {activeInfo.title}
              </h3>
              <p style={{ color: '#ccc', fontSize: '1.1rem', lineHeight: 1.6, marginBottom: '2rem' }}>
                {activeInfo.description}
              </p>
              
              <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderLeft: `4px solid ${activeInfo.color}` }}>
                <h4 style={{ color: 'white', marginBottom: '0.5rem' }}>SIGNAL DETECTED: {activeInfo.signal}</h4>
                <p style={{ color: '#aaa', fontSize: '0.95rem', lineHeight: 1.5, margin: 0 }}>
                  {activeInfo.signalDesc}
                </p>
              </div>
              
              <button 
                onClick={() => setActiveLayer(null)}
                style={{ marginTop: '2rem', background: 'transparent', border: '1px solid #555', color: '#aaa', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: '0.8rem' }}
              >
                CLEAR SELECTION
              </button>
            </div>
          ) : (
            <div className="fade-in">
              <h3 style={{ color: '#ff3333', marginBottom: '1rem' }}>INTERACTIVE VIEW</h3>
              <p style={{ color: '#aaa', fontSize: '1.1rem', lineHeight: 1.6 }}>
                You are looking inside the brain of the Qwen-0.5B model. Each floating sphere represents an individual neuron (parameter), and the lines represent synaptic connections transferring data.
              </p>
              <div style={{ marginTop: '2rem', padding: '1rem', border: '1px dashed #555', textAlign: 'center' }}>
                <p style={{ color: '#fff', fontWeight: 'bold', margin: 0 }}>Click on any glowing layer in the 3D model to dive deep into how it works.</p>
              </div>
            </div>
          )}
        </div>

      </div>
      
      {/* Help Overlay (Bottom Right) */}
      <div style={{ position: 'absolute', bottom: '2rem', right: '2rem', color: '#555', fontSize: '0.8rem', pointerEvents: 'none', textAlign: 'right' }}>
        <strong>CONTROLS</strong><br/>
        Left Click + Drag: Rotate<br/>
        Scroll: Zoom<br/>
        Click Layer: Inspect<br/>
        Click Empty Space: Deselect
      </div>
    </div>
  );
}
