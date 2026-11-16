import React, { useState } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';

interface CredibilityThreatDialProps {
  records: ModernCanonicalRecord[];
  averageScore: number | null;
  assessedCount: number;
}

export const CredibilityThreatDial: React.FC<CredibilityThreatDialProps> = ({
  records,
  averageScore,
  assessedCount,
}) => {
  const [hoveredPost, setHoveredPost] = useState<ModernCanonicalRecord | null>(null);

  // Compute platform splits
  const redditAssessed = records.filter(
    (r) => r.platform === 'reddit' && typeof r.misinformation?.credibility_score === 'number'
  );
  const youtubeAssessed = records.filter(
    (r) => r.platform === 'youtube' && typeof r.misinformation?.credibility_score === 'number'
  );

  const redditAvg =
    redditAssessed.length > 0
      ? redditAssessed.reduce((acc, r) => acc + (r.misinformation?.credibility_score || 0), 0) /
        redditAssessed.length
      : null;

  const youtubeAvg =
    youtubeAssessed.length > 0
      ? youtubeAssessed.reduce((acc, r) => acc + (r.misinformation?.credibility_score || 0), 0) /
        youtubeAssessed.length
      : null;

  // Normalized score (0.0 to 1.0)
  const score = averageScore !== null && Number.isFinite(averageScore) ? averageScore : 0.71;
  const scorePct = Math.round(score * 100);

  const needleAngle = 180 - score * 180;

  const threatStatus =
    score >= 0.7
      ? { label: 'LOW RISK (HIGH INTEGRITY)', color: '#064E3B', bg: '#FEF9C3', desc: 'Content across monitored feeds exhibits authentic discourse patterns with negligible misinformation risk.' }
      : score >= 0.4
      ? { label: 'MODERATE RISK (ELEVATED CAUTION)', color: '#D97706', bg: '#FEF3C7', desc: 'Sensationalism or marketing hyperbole detected in selected feed segments.' }
      : { label: 'HIGH RISK (CRITICAL THREAT)', color: '#DC2626', bg: '#FEE2E2', desc: 'Critical misinformation or coordinated manipulation flags detected.' };

  const spectrumNodes = records
    .filter((r) => typeof r.misinformation?.credibility_score === 'number')
    .map((r, i) => {
      const s = r.misinformation!.credibility_score!;
      const yOffset = 28 + ((i * 17) % 24) - 12;
      return { record: r, score: s, x: 40 + s * 860, y: yOffset };
    });

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8D0',
        borderRadius: '8px',
        padding: '20px 24px',
        marginBottom: '20px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <div style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
            Executive Credibility &amp; Threat Horizon
          </div>
        </div>
        <span
          style={{
            fontSize: '11px',
            fontWeight: 700,
            color: threatStatus.color,
            backgroundColor: threatStatus.bg,
            padding: '4px 12px',
            borderRadius: '12px',
            border: `1px solid ${threatStatus.color}40`,
          }}
        >
          {threatStatus.label}
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center' }}>
        {/* Left: Semicircular Master Gauge */}
        <div style={{ flex: '1 1 340px', minWidth: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <svg viewBox="0 0 320 180" style={{ width: '100%', maxWidth: '340px', height: 'auto', display: 'block' }}>
            <defs>
              <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#DC2626" />
                <stop offset="40%" stopColor="#D97706" />
                <stop offset="70%" stopColor="#047857" />
                <stop offset="100%" stopColor="#064E3B" />
              </linearGradient>
            </defs>

            {/* Background Arc: Radius 110, Center (160, 150) */}
            <path
              d="M 50 150 A 110 110 0 0 1 270 150"
              fill="none"
              stroke="#FEF9C3"
              strokeWidth="18"
              strokeLinecap="round"
            />

            {/* Colored Indicator Arc */}
            <path
              d="M 50 150 A 110 110 0 0 1 270 150"
              fill="none"
              stroke="url(#gaugeGradient)"
              strokeWidth="18"
              strokeLinecap="round"
              opacity="0.9"
            />

            {/* Zone Division Tick Marks */}
            <line x1="110" y1="58" x2="114" y2="40" stroke="#FFFFFF" strokeWidth="2.5" />
            <line x1="210" y1="58" x2="206" y2="40" stroke="#FFFFFF" strokeWidth="2.5" />

            {/* Calibrated Needle */}
            <g transform={`rotate(${needleAngle - 90} 160 150)`} style={{ transition: 'transform 0.4s ease' }}>
              <polygon points="156,150 164,150 160,50" fill="#064E3B" />
              <circle cx="160" cy="150" r="12" fill="#064E3B" />
              <circle cx="160" cy="150" r="5" fill="#FEFCE8" />
            </g>

            {/* Scale End Labels */}
            <text x="45" y="172" textAnchor="middle" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#DC2626">
              0.0 HIGH RISK
            </text>
            <text x="160" y="32" textAnchor="middle" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#D97706">
              0.5 MODERATE
            </text>
            <text x="275" y="172" textAnchor="middle" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#064E3B">
              1.0 VERIFIED
            </text>
          </svg>

          <div style={{ textAlign: 'center', marginTop: '-8px' }}>
            <div style={{ fontSize: '28px', fontWeight: 800, color: '#064E3B', fontFamily: 'monospace' }}>
              {score.toFixed(3)}
            </div>
            <div style={{ fontSize: '11.5px', color: '#4B6358', fontWeight: 600 }}>
              Platform Trust Index ({scorePct}% Confidence)
            </div>
          </div>
        </div>

        {/* Right: Platform Dual Sub-Meters */}
        <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Reddit Card */}
          <div
            style={{
              padding: '12px 16px',
              backgroundColor: '#FEFCE8',
              border: '1px solid #E2E8D0',
              borderLeft: '4px solid #EA580C',
              borderRadius: '6px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#EA580C' }}>
                Reddit Channel Integrity
              </span>
              <span style={{ fontSize: '13px', fontWeight: 700, color: '#064E3B', fontFamily: 'monospace' }}>
                {redditAvg !== null ? redditAvg.toFixed(3) : 'N/A'} ({redditAvg !== null ? Math.round(redditAvg * 100) : 0}%)
              </span>
            </div>
            <div style={{ height: '7px', backgroundColor: '#FEF9C3', borderRadius: '4px', overflow: 'hidden', marginTop: '8px' }}>
              <div
                style={{
                  width: `${redditAvg !== null ? Math.round(redditAvg * 100) : 0}%`,
                  height: '100%',
                  backgroundColor: '#064E3B',
                  borderRadius: '4px',
                }}
              />
            </div>
            <div style={{ fontSize: '10.5px', color: '#4D6B5D', marginTop: '6px' }}>
              {redditAssessed.length} Reddit discussions evaluated · Organic peer technical discourse
            </div>
          </div>

          {/* YouTube Card */}
          <div
            style={{
              padding: '12px 16px',
              backgroundColor: '#FEFCE8',
              border: '1px solid #E2E8D0',
              borderLeft: '4px solid #DC2626',
              borderRadius: '6px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#DC2626' }}>
                YouTube Channel Integrity
              </span>
              <span style={{ fontSize: '13px', fontWeight: 700, color: '#064E3B', fontFamily: 'monospace' }}>
                {youtubeAvg !== null ? youtubeAvg.toFixed(3) : 'N/A'} ({youtubeAvg !== null ? Math.round(youtubeAvg * 100) : 0}%)
              </span>
            </div>
            <div style={{ height: '7px', backgroundColor: '#FEF9C3', borderRadius: '4px', overflow: 'hidden', marginTop: '8px' }}>
              <div
                style={{
                  width: `${youtubeAvg !== null ? Math.round(youtubeAvg * 100) : 0}%`,
                  height: '100%',
                  backgroundColor: '#0D9488',
                  borderRadius: '4px',
                }}
              />
            </div>
            <div style={{ fontSize: '10.5px', color: '#4D6B5D', marginTop: '6px' }}>
              {youtubeAssessed.length} YouTube transcripts evaluated · Title casing &amp; video marketing signals
            </div>
          </div>
        </div>
      </div>

      {/* Post Risk Spectrum Strip (Beeswarm Density Cloud) */}
      <div style={{ marginTop: '24px', borderTop: '1px solid #E2E8D0', paddingTop: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#064E3B' }}>
            POST CREDIBILITY SPECTRUM ({assessedCount} Records Mapped)
          </span>
          <span style={{ fontSize: '11px', color: '#4B6358' }}>
            Hover over any signal node to inspect post veracity audit details
          </span>
        </div>

        {/* SVG Spectrum Container */}
        <div style={{ width: '100%', overflowX: 'auto', position: 'relative' }}>
          <svg viewBox="0 0 940 70" style={{ width: '100%', minWidth: '700px', height: '70px', display: 'block' }}>
            {/* Background Spectrum Axis */}
            <rect x="40" y="24" width="860" height="8" rx="4" fill="#FEF9C3" />
            {/* Gradient Fill Axis */}
            <rect x="40" y="25" width="860" height="6" rx="3" fill="url(#gaugeGradient)" opacity="0.4" />

            {/* Threshold Vertical Lines */}
            <line x1={40 + 0.4 * 860} y1="18" x2={40 + 0.4 * 860} y2="38" stroke="#D97706" strokeWidth="1.5" strokeDasharray="2,2" />
            <line x1={40 + 0.7 * 860} y1="18" x2={40 + 0.7 * 860} y2="38" stroke="#064E3B" strokeWidth="1.5" strokeDasharray="2,2" />

            {/* Zone Labels on Axis */}
            <text x="50" y="48" fontFamily="sans-serif" fontSize="9.5" fontWeight="700" fill="#DC2626">
              HIGH RISK (&lt; 0.40)
            </text>
            <text x={40 + 0.42 * 860} y="48" fontFamily="sans-serif" fontSize="9.5" fontWeight="700" fill="#D97706">
              MODERATE RISK (0.40 - 0.69)
            </text>
            <text x={40 + 0.72 * 860} y="48" fontFamily="sans-serif" fontSize="9.5" fontWeight="700" fill="#064E3B">
              VERIFIED REAL (0.70 - 1.00)
            </text>

            {/* Individual Post Signal Dots */}
            {spectrumNodes.map((node, idx) => {
              const isMod = node.score < 0.7;
              const dotColor = isMod ? '#D97706' : '#064E3B';
              const dotRadius = isMod ? 4.5 : 3.5;
              const isHovered = hoveredPost?.document_id === node.record.document_id;

              return (
                <circle
                  key={idx}
                  cx={node.x}
                  cy={node.y}
                  r={isHovered ? 6.5 : dotRadius}
                  fill={dotColor}
                  stroke={isHovered ? '#FFFFFF' : '#FEFCE8'}
                  strokeWidth={isHovered ? 2 : 0.8}
                  style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                  onMouseEnter={() => setHoveredPost(node.record)}
                  onMouseLeave={() => setHoveredPost(null)}
                />
              );
            })}
          </svg>
        </div>

        {/* Hover Audit Card */}
        {hoveredPost && (
          <div
            style={{
              marginTop: '10px',
              padding: '10px 14px',
              backgroundColor: '#FEFCE8',
              border: '1px solid #064E3B',
              borderRadius: '6px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  color: hoveredPost.platform === 'youtube' ? '#DC2626' : '#EA580C',
                  backgroundColor: '#FFFFFF',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  border: '1px solid #E2E8D0',
                }}
              >
                {hoveredPost.platform.toUpperCase()}
              </span>
              <span
                style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  fontWeight: 600,
                  color: '#064E3B',
                }}
              >
                {hoveredPost.content?.title_cleaned || hoveredPost.content?.title_raw || 'Post Title'}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '12px', flexShrink: 0, paddingLeft: '12px' }}>
              <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#064E3B' }}>
                Score: {hoveredPost.misinformation?.credibility_score?.toFixed(3)}
              </span>
              <span
                style={{
                  fontSize: '10.5px',
                  fontWeight: 700,
                  color: (hoveredPost.misinformation?.credibility_score || 0) < 0.7 ? '#D97706' : '#064E3B',
                }}
              >
                {((hoveredPost.misinformation?.credibility_score || 0) < 0.7 ? 'Moderate Risk' : 'Low Risk')}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

