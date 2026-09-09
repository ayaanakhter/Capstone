import React, { useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

const NUM_NEURONS = 200;
const RADIUS = 4;

function NeuralLayer({ position, color, neuronCount, isActive, isDimmed, onClick }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const positions = useMemo(() => {
    const pos = [];
    for (let i = 0; i < neuronCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * RADIUS;
      const x = r * Math.cos(theta);
      const z = r * Math.sin(theta);
      const y = (Math.random() - 0.5) * 1.5; 
      pos.push(new THREE.Vector3(x, y, z));
    }
    return pos;
  }, [neuronCount]);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    meshRef.current.rotation.y = time * 0.1;
    
    positions.forEach((pos, i) => {
      dummy.position.copy(pos);
      // If hovered or active, make the pulsing more aggressive
      const intensity = (isActive || hovered) ? 0.6 : 0.3;
      const speed = (isActive || hovered) ? 5 : 3;
      const scale = 1 + Math.sin(time * speed + i) * intensity;
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh 
      ref={meshRef} 
      args={[null, null, neuronCount]} 
      position={position}
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer'; }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = 'default'; }}
    >
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial 
        color={hovered ? '#ffffff' : color} 
        emissive={hovered ? '#ffffff' : color} 
        emissiveIntensity={isActive ? 2.0 : (isDimmed ? 0.2 : 1.5)}
        transparent
        opacity={isDimmed ? 0.1 : 0.8}
      />
    </instancedMesh>
  );
}

function SynapseConnections({ layer1Y, layer2Y, color, isDimmed }) {
  const lineRef = useRef();

  const geometry = useMemo(() => {
    const points = [];
    for (let i = 0; i < 300; i++) {
      const t1 = Math.random() * Math.PI * 2;
      const r1 = Math.sqrt(Math.random()) * RADIUS;
      points.push(new THREE.Vector3(r1 * Math.cos(t1), layer1Y + (Math.random()-0.5), r1 * Math.sin(t1)));
      
      const t2 = Math.random() * Math.PI * 2;
      const r2 = Math.sqrt(Math.random()) * RADIUS;
      points.push(new THREE.Vector3(r2 * Math.cos(t2), layer2Y + (Math.random()-0.5), r2 * Math.sin(t2)));
    }
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [layer1Y, layer2Y]);

  useFrame((state) => {
    lineRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
  });

  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial 
        color={color} 
        transparent 
        opacity={isDimmed ? 0.02 : 0.15} 
        blending={THREE.AdditiveBlending} 
      />
    </lineSegments>
  );
}

export default function Architecture3D({ activeLayer, onSelectLayer }) {
  return (
    <Canvas camera={{ position: [12, 5, 12], fov: 60 }} onPointerMissed={() => onSelectLayer(null)}>
      <color attach="background" args={['#050505']} />
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={2} color="#ffffff" />
      <pointLight position={[-10, -10, -10]} intensity={1} color="#ff3333" />
      
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      <OrbitControls enablePan={false} autoRotate={activeLayer === null} autoRotateSpeed={0.5} maxDistance={25} minDistance={5} />

      <group position={[0, -2, 0]}>
        
        {/* Layer 0: Inputs */}
        <NeuralLayer 
          position={[0, -5, 0]} 
          color="#3366ff" 
          neuronCount={NUM_NEURONS} 
          isActive={activeLayer === 0}
          isDimmed={activeLayer !== null && activeLayer !== 0}
          onClick={() => onSelectLayer(0)}
        />
        
        <SynapseConnections 
          layer1Y={-5} layer2Y={0} color="#7733ff" 
          isDimmed={activeLayer !== null && activeLayer !== 0 && activeLayer !== 1} 
        />

        {/* Layer 1: Hidden (MLP/Attention) */}
        <NeuralLayer 
          position={[0, 0, 0]} 
          color="#aa00ff" 
          neuronCount={NUM_NEURONS} 
          isActive={activeLayer === 1}
          isDimmed={activeLayer !== null && activeLayer !== 1}
          onClick={() => onSelectLayer(1)}
        />

        <SynapseConnections 
          layer1Y={0} layer2Y={5} color="#ff3333" 
          isDimmed={activeLayer !== null && activeLayer !== 1 && activeLayer !== 2} 
        />

        {/* Layer 2: Output / LM Head */}
        <NeuralLayer 
          position={[0, 5, 0]} 
          color="#ff3333" 
          neuronCount={NUM_NEURONS} 
          isActive={activeLayer === 2}
          isDimmed={activeLayer !== null && activeLayer !== 2}
          onClick={() => onSelectLayer(2)}
        />
      </group>
    </Canvas>
  );
}
