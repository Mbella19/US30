import React, { useState, useMemo } from 'react';
import useVisualizationData from '../hooks/useVisualizationData';

// Model architecture - US30 with mean reversion features
const MODEL_CONFIG = {
  inputSize: 145,
  actor: { layers: [256, 256], output: { direction: 3, size: 4 } },
  critic: { layers: [512, 512, 256], output: 1 },
  actions: ['FLAT', 'LONG', 'SHORT'],
  sizes: ['0.5x', '1.0x', '1.5x', '2.0x']
};

// Feature groups for input visualization
const FEATURE_GROUPS = {
  'Market Context': { start: 0, count: 32, color: '#8B7355' },
  'Volatility Metrics': { start: 32, count: 24, color: '#6B8E6B' },
  'Trend Indicators': { start: 56, count: 28, color: '#7B6B8B' },
  'Momentum Signals': { start: 84, count: 24, color: '#8B6B6B' },
  'Session & Time': { start: 108, count: 20, color: '#6B7B8B' },
  'Position State': { start: 128, count: 17, color: '#8B8B6B' }
};

// Utility for generating deterministic pseudo-random weights
const seededRandom = (seed) => {
  const x = Math.sin(seed * 9999) * 10000;
  return x - Math.floor(x);
};

// Normalize activation for visualization (handle large values)
const normalizeActivation = (act) => {
  const raw = Math.abs(act || 0);
  return raw > 1 ? Math.tanh(raw * 0.5) : raw;
};

// Neuron component with activation visualization
const Neuron = ({ x, y, activation, radius = 4, isOutput = false, label = '' }) => {
  const intensity = normalizeActivation(activation);
  const isPositive = (activation || 0) >= 0;

  const fillColor = isOutput
    ? (isPositive ? `rgba(139, 115, 85, ${0.3 + intensity * 0.7})` : `rgba(107, 85, 85, ${0.3 + intensity * 0.7})`)
    : `rgba(196, 181, 157, ${0.15 + intensity * 0.6})`;

  const glowIntensity = intensity * 6;

  return (
    <g>
      {intensity > 0.4 && (
        <circle
          cx={x}
          cy={y}
          r={radius + glowIntensity}
          fill={`rgba(196, 181, 157, ${intensity * 0.12})`}
          style={{ transition: 'all 0.3s ease-out' }}
        />
      )}
      <circle
        cx={x}
        cy={y}
        r={radius}
        fill={fillColor}
        stroke={intensity > 0.5 ? '#D4C4A8' : '#3D3D3D'}
        strokeWidth={intensity > 0.5 ? 1.5 : 0.5}
        style={{ transition: 'all 0.2s ease-out' }}
      />
      {label && (
        <text x={x} y={y + radius + 12} textAnchor="middle" fill="#8B8B8B" fontSize="9" fontFamily="'IBM Plex Mono', monospace">
          {label}
        </text>
      )}
    </g>
  );
};

// Connection component with weight visualization
const Connection = ({ x1, y1, x2, y2, weight, activation }) => {
  const rawIntensity = Math.abs((weight || 0) * normalizeActivation(activation));
  const intensity = Math.min(1, rawIntensity);
  const isPositive = (weight || 0) >= 0;

  const strokeColor = isPositive
    ? `rgba(139, 115, 85, ${Math.min(0.5, 0.03 + intensity * 0.4)})`
    : `rgba(85, 85, 107, ${Math.min(0.5, 0.03 + intensity * 0.4)})`;

  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={strokeColor}
      strokeWidth={Math.max(0.3, Math.min(1.5, intensity * 1.5))}
      style={{ transition: 'all 0.3s ease-out' }}
    />
  );
};

// Layer visualization component
const Layer = ({ neurons, x, height, startY, label, activations, maxVisible = 16 }) => {
  const visibleCount = Math.min(neurons, maxVisible);
  const spacing = Math.min(height / (visibleCount + 1), 14);
  const actualStartY = startY + (height - spacing * (visibleCount - 1)) / 2;

  return (
    <g>
      <text x={x} y={startY - 20} textAnchor="middle" fill="#6B6B6B" fontSize="10" fontFamily="'IBM Plex Mono', monospace" fontWeight="500">
        {label}
      </text>
      <text x={x} y={startY - 6} textAnchor="middle" fill="#4A4A4A" fontSize="8" fontFamily="'IBM Plex Mono', monospace">
        {neurons} units
      </text>
      {Array.from({ length: visibleCount }, (_, i) => {
        const neuronIndex = neurons > maxVisible
          ? Math.floor(i * neurons / visibleCount)
          : i;
        return (
          <Neuron
            key={i}
            x={x}
            y={actualStartY + i * spacing}
            activation={activations?.[neuronIndex] || 0}
            radius={neurons > 100 ? 3 : 4}
          />
        );
      })}
      {neurons > maxVisible && (
        <text x={x} y={actualStartY + visibleCount * spacing + 15} textAnchor="middle" fill="#4A4A4A" fontSize="8" fontFamily="'IBM Plex Mono', monospace" fontStyle="italic">
          ({neurons - maxVisible} more)
        </text>
      )}
    </g>
  );
};

// Input features grouped visualization
const InputFeatures = ({ x, height, startY, activations }) => {
  const groups = Object.entries(FEATURE_GROUPS);
  const groupHeight = height / groups.length;

  return (
    <g>
      <text x={x - 30} y={startY - 20} textAnchor="middle" fill="#6B6B6B" fontSize="10" fontFamily="'IBM Plex Mono', monospace" fontWeight="500">
        INPUT
      </text>
      <text x={x - 30} y={startY - 6} textAnchor="middle" fill="#4A4A4A" fontSize="8" fontFamily="'IBM Plex Mono', monospace">
        145 features
      </text>

      {groups.map(([name, config], groupIdx) => {
        const groupY = startY + groupIdx * groupHeight;

        return (
          <g key={name}>
            <text x={x - 80} y={groupY + groupHeight / 2} textAnchor="end" fill={config.color} fontSize="8" fontFamily="'IBM Plex Mono', monospace" opacity="0.8">
              {name}
            </text>
            {Array.from({ length: Math.min(config.count, 16) }, (_, i) => {
              const row = Math.floor(i / 4);
              const col = i % 4;
              const idx = config.start + i;
              return (
                <Neuron
                  key={i}
                  x={x - 50 + col * 12}
                  y={groupY + 8 + row * 12}
                  activation={activations?.[idx] || 0}
                  radius={4}
                />
              );
            })}
          </g>
        );
      })}
    </g>
  );
};

// Stats panel component
const StatsPanel = ({ title, children }) => (
  <div style={{
    background: 'linear-gradient(180deg, #1A1A1A 0%, #141414 100%)',
    border: '1px solid #2A2A2A',
    borderRadius: '6px',
    padding: '16px',
    marginBottom: '12px'
  }}>
    <div style={{
      fontSize: '10px',
      color: '#6B6B6B',
      fontFamily: "'IBM Plex Mono', monospace",
      textTransform: 'uppercase',
      letterSpacing: '1px',
      marginBottom: '12px',
      borderBottom: '1px solid #2A2A2A',
      paddingBottom: '8px'
    }}>
      {title}
    </div>
    {children}
  </div>
);

const StatRow = ({ label, value, highlight = false, color = null }) => (
  <div style={{
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
    fontSize: '11px',
    fontFamily: "'IBM Plex Mono', monospace"
  }}>
    <span style={{ color: '#5A5A5A' }}>{label}</span>
    <span style={{ color: color || (highlight ? '#C4B59D' : '#8B8B8B'), fontWeight: highlight ? '600' : '400' }}>
      {value}
    </span>
  </div>
);

// Main visualization component
const NeuralVisualizer = () => {
  const { data, connected, messageCount } = useVisualizationData();
  const [selectedView, setSelectedView] = useState('combined');

  // Extract data from WebSocket
  const market = data?.market || {};
  const position = data?.position || {};
  const analystData = data?.analyst || {};
  const agentData = data?.agent || {};
  const decision = data?.decision || {};
  const features = data?.features || {};

  // Get activations from real data
  const inputActivations = useMemo(() => {
    // Build input activations from observation or generate placeholder
    return Array(145).fill(0).map((_, i) => {
      // Use feature data if available
      const feat5m = features['5m'] || {};
      const keys = Object.keys(feat5m);
      if (i < keys.length) {
        const val = feat5m[keys[i]] || 0;
        return normalizeActivation(val);
      }
      return 0;
    });
  }, [features]);

  const actorL1 = agentData?.actor_layers?.layer_0 || [];
  const actorL2 = agentData?.actor_layers?.layer_1 || [];
  const criticL1 = agentData?.critic_layers?.layer_0 || [];
  const criticL2 = agentData?.critic_layers?.layer_1 || [];
  const criticL3 = agentData?.critic_layers?.layer_2 || [];
  const actionProbs = agentData?.action_probs || { flat: 0.33, long: 0.33, short: 0.34 };
  const valueEstimate = agentData?.value_estimate || 0;

  const svgWidth = 900;
  const svgHeight = 500;
  const startY = 60;

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(145deg, #0D0D0D 0%, #111111 50%, #0A0A0A 100%)',
      color: '#E8E8E8',
      fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      padding: '0',
      overflow: 'hidden'
    }}>
      {/* Subtle texture overlay */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
        opacity: 0.03,
        pointerEvents: 'none',
        zIndex: 1
      }} />

      {/* Header */}
      <header style={{
        padding: '24px 40px',
        borderBottom: '1px solid #1A1A1A',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        position: 'relative',
        zIndex: 10,
        background: 'rgba(13, 13, 13, 0.95)',
        backdropFilter: 'blur(10px)'
      }}>
        <div>
          <h1 style={{
            margin: 0,
            fontSize: '20px',
            fontFamily: "'IBM Plex Serif', Georgia, serif",
            fontWeight: '500',
            color: '#C4B59D',
            letterSpacing: '-0.5px'
          }}>
            Neural Architecture Visualizer - US30
          </h1>
          <p style={{
            margin: '4px 0 0 0',
            fontSize: '11px',
            color: '#5A5A5A',
            fontFamily: "'IBM Plex Mono', monospace",
            letterSpacing: '0.5px'
          }}>
            PPO SNIPER AGENT · LIVE ACTIVATIONS · ACTOR-CRITIC ARCHITECTURE
          </p>
        </div>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{
            display: 'flex',
            background: '#1A1A1A',
            borderRadius: '4px',
            padding: '2px'
          }}>
            {['combined', 'actor', 'critic'].map(view => (
              <button
                key={view}
                onClick={() => setSelectedView(view)}
                style={{
                  padding: '6px 14px',
                  border: 'none',
                  background: selectedView === view ? '#2A2A2A' : 'transparent',
                  color: selectedView === view ? '#C4B59D' : '#5A5A5A',
                  fontSize: '10px',
                  fontFamily: "'IBM Plex Mono', monospace",
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                  borderRadius: '3px',
                  transition: 'all 0.2s ease'
                }}
              >
                {view}
              </button>
            ))}
          </div>

          <div style={{
            padding: '8px 16px',
            border: '1px solid #2A2A2A',
            background: connected ? '#1A1A1A' : 'transparent',
            color: connected ? '#6B8B6B' : '#8B6B6B',
            fontSize: '10px',
            fontFamily: "'IBM Plex Mono', monospace",
            textTransform: 'uppercase',
            borderRadius: '4px',
          }}>
            {connected ? `● LIVE (${messageCount})` : '○ DISCONNECTED'}
          </div>
        </div>
      </header>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '240px 1fr 200px',
        gap: '0',
        height: 'calc(100vh - 85px)',
        position: 'relative',
        zIndex: 10
      }}>
        {/* Left Panel - Model Info */}
        <aside style={{
          padding: '20px',
          borderRight: '1px solid #1A1A1A',
          overflowY: 'auto',
          background: 'rgba(10, 10, 10, 0.5)'
        }}>
          <StatsPanel title="Market State">
            <StatRow label="Price" value={market.price?.toFixed(2) || '-'} highlight />
            <StatRow label="ATR" value={market.atr?.toFixed(2) || '-'} />
            <StatRow label="Spread" value={`${market.spread?.toFixed(1) || '-'} pips`} />
          </StatsPanel>

          <StatsPanel title="Position">
            <StatRow
              label="Type"
              value={position.type_name || 'FLAT'}
              color={position.type_name === 'LONG' ? '#6B8B6B' : position.type_name === 'SHORT' ? '#8B6B6B' : '#6B6B6B'}
            />
            <StatRow label="Volume" value={position.volume?.toFixed(2) || '0.00'} />
            <StatRow label="Entry" value={position.entry_price?.toFixed(2) || '-'} />
            <StatRow
              label="P&L"
              value={`$${position.unrealized_pnl?.toFixed(2) || '0.00'}`}
              color={(position.unrealized_pnl || 0) >= 0 ? '#6B8B6B' : '#8B6B6B'}
            />
          </StatsPanel>

          <StatsPanel title="Architecture">
            <StatRow label="Input Features" value="145" highlight />
            <StatRow label="Actor Hidden" value="256 × 256" />
            <StatRow label="Critic Hidden" value="512 × 512 × 256" />
            <StatRow label="Total Parameters" value="~890K" />
          </StatsPanel>

          <StatsPanel title="Input Features">
            {Object.entries(FEATURE_GROUPS).map(([name, config]) => (
              <div key={name} style={{ marginBottom: '8px' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '4px'
                }}>
                  <span style={{ fontSize: '10px', color: config.color, fontFamily: "'IBM Plex Mono', monospace" }}>
                    {name}
                  </span>
                  <span style={{ fontSize: '9px', color: '#4A4A4A', fontFamily: "'IBM Plex Mono', monospace" }}>
                    {config.count}
                  </span>
                </div>
                <div style={{
                  height: '2px',
                  background: '#1A1A1A',
                  borderRadius: '1px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${(config.count / 145) * 100}%`,
                    height: '100%',
                    background: config.color,
                    opacity: 0.5
                  }} />
                </div>
              </div>
            ))}
          </StatsPanel>
        </aside>

        {/* Main Visualization */}
        <main style={{
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}>
          {/* Connection status */}
          {!connected && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              zIndex: 100,
            }}>
              <div style={{
                fontSize: '14px',
                color: '#8B6B6B',
                fontFamily: "'IBM Plex Mono', monospace",
                marginBottom: '8px',
              }}>
                Waiting for connection...
              </div>
              <div style={{
                fontSize: '10px',
                color: '#5A5A5A',
                fontFamily: "'IBM Plex Mono', monospace",
              }}>
                Start the bridge: python scripts/core/run_visualizer.py
              </div>
            </div>
          )}

          <svg width={svgWidth} height={svgHeight} style={{ overflow: 'visible', opacity: connected ? 1 : 0.3 }}>
            {/* Grid lines for depth */}
            {Array.from({ length: 20 }, (_, i) => (
              <line
                key={i}
                x1={0}
                y1={startY + i * 20}
                x2={svgWidth}
                y2={startY + i * 20}
                stroke="#151515"
                strokeWidth="0.5"
              />
            ))}

            {/* Actor Network (top) */}
            {(selectedView === 'combined' || selectedView === 'actor') && (
              <g transform={selectedView === 'actor' ? 'translate(0, 80)' : ''}>
                <text x={450} y={25} textAnchor="middle" fill="#8B7355" fontSize="11" fontFamily="'IBM Plex Mono', monospace" fontWeight="600" letterSpacing="2">
                  ACTOR NETWORK (POLICY)
                </text>

                {/* Input to Actor L1 connections */}
                <g opacity="0.4">
                  {Array.from({ length: 20 }, (_, i) => {
                    const inputIdx = Math.floor(i * 145 / 20);
                    const fromY = startY + 40 + (i / 20) * 160;
                    return Array.from({ length: 8 }, (_, j) => {
                      const toIdx = Math.floor(j * 256 / 8);
                      const toY = startY + 50 + (j / 8) * 140;
                      return (
                        <Connection
                          key={`${i}-${j}`}
                          x1={180}
                          y1={fromY}
                          x2={340}
                          y2={toY}
                          weight={seededRandom(inputIdx * 256 + toIdx)}
                          activation={inputActivations[inputIdx]}
                        />
                      );
                    });
                  })}
                </g>

                {/* Actor L1 to L2 connections */}
                <g opacity="0.5">
                  {Array.from({ length: 12 }, (_, i) => {
                    const fromIdx = Math.floor(i * 256 / 12);
                    const fromY = startY + 50 + (i / 12) * 140;
                    return Array.from({ length: 12 }, (_, j) => {
                      const toIdx = Math.floor(j * 256 / 12);
                      const toY = startY + 50 + (j / 12) * 140;
                      return (
                        <Connection
                          key={`a1-${i}-${j}`}
                          x1={340}
                          y1={fromY}
                          x2={480}
                          y2={toY}
                          weight={seededRandom(2000 + fromIdx * 256 + toIdx)}
                          activation={actorL1[fromIdx] || 0}
                        />
                      );
                    });
                  })}
                </g>

                {/* Actor L2 to Output */}
                <g opacity="0.6">
                  {Array.from({ length: 12 }, (_, i) => {
                    const fromIdx = Math.floor(i * 256 / 12);
                    const fromY = startY + 50 + (i / 12) * 140;
                    return Array.from({ length: 3 }, (_, j) => {
                      const toY = startY + 80 + j * 40;
                      return (
                        <Connection
                          key={`a2-${i}-${j}`}
                          x1={480}
                          y1={fromY}
                          x2={600}
                          y2={toY}
                          weight={seededRandom(5000 + fromIdx * 3 + j)}
                          activation={actorL2[fromIdx] || 0}
                        />
                      );
                    });
                  })}
                </g>

                {/* Input Features */}
                <InputFeatures
                  x={180}
                  height={200}
                  startY={startY + 30}
                  activations={inputActivations}
                />

                {/* Actor Hidden Layer 1 */}
                <Layer
                  neurons={256}
                  x={340}
                  height={180}
                  startY={startY + 30}
                  label="HIDDEN 1"
                  activations={actorL1}
                  maxVisible={16}
                />

                {/* Actor Hidden Layer 2 */}
                <Layer
                  neurons={256}
                  x={480}
                  height={180}
                  startY={startY + 30}
                  label="HIDDEN 2"
                  activations={actorL2}
                  maxVisible={16}
                />

                {/* Action Output Neurons */}
                <g>
                  <text x={620} y={startY + 10} textAnchor="middle" fill="#6B6B6B" fontSize="10" fontFamily="'IBM Plex Mono', monospace" fontWeight="500">
                    ACTIONS
                  </text>
                  {MODEL_CONFIG.actions.map((action, i) => {
                    const isSelected = decision.action_name === action;
                    const prob = action === 'FLAT' ? actionProbs.flat : action === 'LONG' ? actionProbs.long : actionProbs.short;
                    const y = startY + 60 + i * 45;
                    return (
                      <g key={action}>
                        <circle
                          cx={620}
                          cy={y}
                          r={isSelected ? 12 : 8}
                          fill={isSelected ?
                            (action === 'LONG' ? 'rgba(107, 139, 107, 0.8)' :
                             action === 'SHORT' ? 'rgba(139, 107, 107, 0.8)' :
                             'rgba(107, 107, 107, 0.8)') :
                            '#1A1A1A'}
                          stroke={isSelected ? '#C4B59D' : '#3D3D3D'}
                          strokeWidth={isSelected ? 2 : 1}
                          style={{ transition: 'all 0.3s ease' }}
                        />
                        {isSelected && (
                          <circle
                            cx={620}
                            cy={y}
                            r={18}
                            fill="none"
                            stroke="#C4B59D"
                            strokeWidth={1}
                            opacity={0.3}
                          />
                        )}
                        <text
                          x={648}
                          y={y + 4}
                          fill={isSelected ? '#C4B59D' : '#5A5A5A'}
                          fontSize="10"
                          fontFamily="'IBM Plex Mono', monospace"
                          fontWeight={isSelected ? '600' : '400'}
                        >
                          {action}
                        </text>
                      </g>
                    );
                  })}
                </g>

                {/* Confidence Bar */}
                <g>
                  <rect x={710} y={startY + 40} width={80} height={100} rx={4} fill="#1A1A1A" stroke="#2A2A2A" />
                  <text x={750} y={startY + 58} textAnchor="middle" fill="#5A5A5A" fontSize="8" fontFamily="'IBM Plex Mono', monospace">
                    CONFIDENCE
                  </text>
                  <rect x={720} y={startY + 70} width={60} height={8} rx={2} fill="#252525" />
                  <rect
                    x={720}
                    y={startY + 70}
                    width={60 * (decision.confidence || 0)}
                    height={8}
                    rx={2}
                    fill="#8B7355"
                    style={{ transition: 'width 0.3s ease' }}
                  />
                  <text x={750} y={startY + 98} textAnchor="middle" fill="#C4B59D" fontSize="14" fontFamily="'IBM Plex Mono', monospace" fontWeight="600">
                    {((decision.confidence || 0) * 100).toFixed(1)}%
                  </text>
                  <text x={750} y={startY + 120} textAnchor="middle" fill="#5A5A5A" fontSize="9" fontFamily="'IBM Plex Mono', monospace">
                    Size: {decision.size_name || '-'}
                  </text>
                </g>
              </g>
            )}

            {/* Critic Network (bottom) */}
            {(selectedView === 'combined' || selectedView === 'critic') && (
              <g transform={selectedView === 'combined' ? 'translate(0, 280)' : 'translate(0, 80)'}>
                <text x={450} y={25} textAnchor="middle" fill="#6B7B8B" fontSize="11" fontFamily="'IBM Plex Mono', monospace" fontWeight="600" letterSpacing="2">
                  CRITIC NETWORK (VALUE)
                </text>

                {/* Input to Critic L1 */}
                <g opacity="0.3">
                  {Array.from({ length: 15 }, (_, i) => {
                    const inputIdx = Math.floor(i * 145 / 15);
                    const fromY = startY + 40 + (i / 15) * 140;
                    return Array.from({ length: 10 }, (_, j) => {
                      const toIdx = Math.floor(j * 512 / 10);
                      const toY = startY + 45 + (j / 10) * 130;
                      return (
                        <Connection
                          key={`c0-${i}-${j}`}
                          x1={180}
                          y1={fromY}
                          x2={300}
                          y2={toY}
                          weight={seededRandom(1000 + inputIdx * 512 + toIdx)}
                          activation={inputActivations[inputIdx]}
                        />
                      );
                    });
                  })}
                </g>

                {/* Input Layer for Critic */}
                <Layer
                  neurons={145}
                  x={180}
                  height={180}
                  startY={startY + 30}
                  label="INPUT"
                  activations={inputActivations}
                  maxVisible={16}
                />

                {/* Critic L1 to L2 */}
                <g opacity="0.4">
                  {Array.from({ length: 10 }, (_, i) => {
                    const fromIdx = Math.floor(i * 512 / 10);
                    const fromY = startY + 45 + (i / 10) * 130;
                    return Array.from({ length: 10 }, (_, j) => {
                      const toIdx = Math.floor(j * 512 / 10);
                      const toY = startY + 45 + (j / 10) * 130;
                      return (
                        <Connection
                          key={`c1-${i}-${j}`}
                          x1={300}
                          y1={fromY}
                          x2={420}
                          y2={toY}
                          weight={seededRandom(3000 + fromIdx * 512 + toIdx)}
                          activation={criticL1[fromIdx] || 0}
                        />
                      );
                    });
                  })}
                </g>

                {/* Critic L2 to L3 */}
                <g opacity="0.5">
                  {Array.from({ length: 10 }, (_, i) => {
                    const fromIdx = Math.floor(i * 512 / 10);
                    const fromY = startY + 45 + (i / 10) * 130;
                    return Array.from({ length: 8 }, (_, j) => {
                      const toIdx = Math.floor(j * 256 / 8);
                      const toY = startY + 50 + (j / 8) * 120;
                      return (
                        <Connection
                          key={`c2-${i}-${j}`}
                          x1={420}
                          y1={fromY}
                          x2={540}
                          y2={toY}
                          weight={seededRandom(4000 + fromIdx * 256 + toIdx)}
                          activation={criticL2[fromIdx] || 0}
                        />
                      );
                    });
                  })}
                </g>

                {/* Critic L3 to Value */}
                <g opacity="0.6">
                  {Array.from({ length: 8 }, (_, i) => {
                    const fromIdx = Math.floor(i * 256 / 8);
                    const fromY = startY + 50 + (i / 8) * 120;
                    return (
                      <Connection
                        key={`c3-${i}`}
                        x1={540}
                        y1={fromY}
                        x2={660}
                        y2={startY + 100}
                        weight={seededRandom(6000 + fromIdx)}
                        activation={criticL3[fromIdx] || 0}
                      />
                    );
                  })}
                </g>

                {/* Critic Hidden Layer 1 */}
                <Layer
                  neurons={512}
                  x={300}
                  height={160}
                  startY={startY + 30}
                  label="HIDDEN 1"
                  activations={criticL1}
                  maxVisible={14}
                />

                {/* Critic Hidden Layer 2 */}
                <Layer
                  neurons={512}
                  x={420}
                  height={160}
                  startY={startY + 30}
                  label="HIDDEN 2"
                  activations={criticL2}
                  maxVisible={14}
                />

                {/* Critic Hidden Layer 3 */}
                <Layer
                  neurons={256}
                  x={540}
                  height={150}
                  startY={startY + 35}
                  label="HIDDEN 3"
                  activations={criticL3}
                  maxVisible={12}
                />

                {/* Value Output */}
                <g>
                  <rect x={630} y={startY + 50} width={100} height={80} rx={4} fill="#1A1A1A" stroke="#2A2A2A" />
                  <text x={680} y={startY + 72} textAnchor="middle" fill="#5A5A5A" fontSize="9" fontFamily="'IBM Plex Mono', monospace">
                    STATE VALUE
                  </text>
                  <text
                    x={680}
                    y={startY + 100}
                    textAnchor="middle"
                    fill={valueEstimate >= 0 ? '#7B9B7B' : '#9B7B7B'}
                    fontSize="18"
                    fontFamily="'IBM Plex Mono', monospace"
                    fontWeight="600"
                  >
                    {valueEstimate >= 0 ? '+' : ''}{valueEstimate.toFixed(4)}
                  </text>
                  <text x={680} y={startY + 118} textAnchor="middle" fill="#4A4A4A" fontSize="8" fontFamily="'IBM Plex Mono', monospace">
                    Expected Return
                  </text>
                </g>
              </g>
            )}
          </svg>

          {/* Legend */}
          <div style={{
            display: 'flex',
            gap: '24px',
            marginTop: '16px',
            padding: '12px 20px',
            background: '#141414',
            borderRadius: '4px',
            border: '1px solid #1A1A1A'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'rgba(196, 181, 157, 0.8)' }} />
              <span style={{ fontSize: '9px', color: '#6B6B6B', fontFamily: "'IBM Plex Mono', monospace" }}>Active Neuron</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#2A2A2A', border: '1px solid #3D3D3D' }} />
              <span style={{ fontSize: '9px', color: '#6B6B6B', fontFamily: "'IBM Plex Mono', monospace" }}>Inactive Neuron</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '20px', height: '2px', background: 'rgba(139, 115, 85, 0.5)' }} />
              <span style={{ fontSize: '9px', color: '#6B6B6B', fontFamily: "'IBM Plex Mono', monospace" }}>Positive Weight</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '20px', height: '2px', background: 'rgba(85, 85, 107, 0.5)' }} />
              <span style={{ fontSize: '9px', color: '#6B6B6B', fontFamily: "'IBM Plex Mono', monospace" }}>Negative Weight</span>
            </div>
          </div>
        </main>

        {/* Right Panel - Live Activity */}
        <aside style={{
          padding: '20px',
          borderLeft: '1px solid #1A1A1A',
          overflowY: 'auto',
          background: 'rgba(10, 10, 10, 0.5)'
        }}>
          <StatsPanel title="Current Decision">
            <div style={{
              textAlign: 'center',
              padding: '16px 0',
              marginBottom: '12px'
            }}>
              <div style={{
                fontSize: '28px',
                fontFamily: "'IBM Plex Serif', serif",
                fontWeight: '600',
                color: decision.action_name === 'LONG' ? '#6B8B6B' :
                       decision.action_name === 'SHORT' ? '#8B6B6B' : '#6B6B6B',
                marginBottom: '8px'
              }}>
                {decision.action_name || 'FLAT'}
              </div>
              <div style={{
                fontSize: '10px',
                color: '#5A5A5A',
                fontFamily: "'IBM Plex Mono', monospace"
              }}>
                {((decision.confidence || 0) * 100).toFixed(1)}% confidence
              </div>
            </div>
            <StatRow label="Position Size" value={decision.size_name || '-'} />
            <StatRow label="Reason" value={decision.reason || '-'} />
            <StatRow label="Value Estimate" value={valueEstimate.toFixed(4)} highlight />
          </StatsPanel>

          <StatsPanel title="Analyst Output">
            <StatRow label="P(Up)" value={`${((analystData.probabilities?.p_up || 0.5) * 100).toFixed(1)}%`} highlight />
            <StatRow label="P(Down)" value={`${((analystData.probabilities?.p_down || 0.5) * 100).toFixed(1)}%`} />
            <StatRow label="Confidence" value={`${((analystData.confidence || 0.5) * 100).toFixed(1)}%`} />
            <StatRow label="Edge" value={(analystData.edge || 0).toFixed(3)} />
          </StatsPanel>

          <StatsPanel title="Network Activity">
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '9px', color: '#5A5A5A', marginBottom: '4px', fontFamily: "'IBM Plex Mono', monospace" }}>
                Actor Hidden ({actorL2.length})
              </div>
              <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap' }}>
                {actorL2.slice(0, 30).map((act, i) => (
                  <div
                    key={i}
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '1px',
                      background: `rgba(139, 115, 85, ${0.2 + normalizeActivation(act) * 0.8})`,
                      transition: 'background 0.3s ease'
                    }}
                  />
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '9px', color: '#5A5A5A', marginBottom: '4px', fontFamily: "'IBM Plex Mono', monospace" }}>
                Critic Hidden ({criticL3.length})
              </div>
              <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap' }}>
                {criticL3.slice(0, 30).map((act, i) => (
                  <div
                    key={i}
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '1px',
                      background: `rgba(107, 123, 139, ${0.2 + normalizeActivation(act) * 0.8})`,
                      transition: 'background 0.3s ease'
                    }}
                  />
                ))}
              </div>
            </div>
          </StatsPanel>

          <StatsPanel title="Action Probabilities">
            {MODEL_CONFIG.actions.map(action => {
              const prob = action === 'FLAT' ? actionProbs.flat : action === 'LONG' ? actionProbs.long : actionProbs.short;
              return (
                <div key={action} style={{ marginBottom: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '10px', color: '#5A5A5A', fontFamily: "'IBM Plex Mono', monospace" }}>{action}</span>
                    <span style={{ fontSize: '10px', color: '#8B8B8B', fontFamily: "'IBM Plex Mono', monospace" }}>{((prob || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: '4px', background: '#1A1A1A', borderRadius: '2px' }}>
                    <div style={{
                      width: `${(prob || 0) * 100}%`,
                      height: '100%',
                      background: action === 'LONG' ? '#6B8B6B' : action === 'SHORT' ? '#8B6B6B' : '#6B6B6B',
                      borderRadius: '2px',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                </div>
              );
            })}
          </StatsPanel>
        </aside>
      </div>
    </div>
  );
};

export default NeuralVisualizer;
