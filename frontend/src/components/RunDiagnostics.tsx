import React, { useState } from 'react';
import type { DatasetSummary } from '../types/dataset';
import { StatusBadge } from './StatusBadge';

interface RunDiagnosticsProps {
  dataset: DatasetSummary;
}

const STANDARD_ARTIFACTS = [
  'consolidated_intelligence.json',
  'sentiment_analysis_results.json',
  'misinformation_analysis_results.json',
  'graph_analysis_results.json',
  'topic_modeling_results.json',
];

export const RunDiagnostics: React.FC<RunDiagnosticsProps> = ({ dataset }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const isModern = dataset.type === 'modern';
  const stageEntries = Object.entries(dataset.stages);

  return (
    <div className="run-diagnostics-container" style={{ margin: '12px 0 20px 0', border: '1px solid #E2E8F0', borderRadius: '6px', backgroundColor: '#FFFFFF', overflow: 'hidden' }}>
      {/* Top Banner / Toggle Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 16px',
          backgroundColor: '#F8FAFC',
          cursor: 'pointer',
          borderBottom: isOpen ? '1px solid #E2E8F0' : 'none',
          userSelect: 'none',
        }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}>
            Pipeline Health:
          </span>
          <StatusBadge status={dataset.status} />
          <span style={{ fontSize: '12px', color: '#64748B' }}>
            {dataset.available_artifacts.length}/{STANDARD_ARTIFACTS.length} Analytical Datasets Verified
          </span>
        </div>

        <button
          type="button"
          style={{
            background: 'none',
            border: 'none',
            color: '#2563EB',
            fontSize: '13px',
            fontWeight: 500,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
          onClick={(e) => {
            e.stopPropagation();
            setIsOpen(!isOpen);
          }}
        >
          {isOpen ? '▲ Hide Diagnostics' : '▼ View Diagnostics & Stage Matrix'}
        </button>
      </div>

      {/* Collapsible Content */}
      {isOpen && (
        <div style={{ padding: '20px 18px', backgroundColor: '#FAFAFA' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '20px' }}>
            {/* Run Identity Box */}
            <div style={{ backgroundColor: '#FFFFFF', padding: '14px', borderRadius: '6px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '10px' }}>
                Run Identity
              </div>
              <div style={{ fontSize: '12px', color: '#334155', lineHeight: '1.6' }}>
                <div><strong>Identifier:</strong> <span className="font-mono">{dataset.id}</span></div>
                <div><strong>Contract Type:</strong> {isModern ? 'Modern Execution Run (Schema-v2)' : 'Historical Baseline'}</div>
                <div><strong>Timestamp UTC:</strong> {dataset.generated_at || 'Not recorded'}</div>
              </div>
            </div>

            {/* Artifact Inventory Box */}
            <div style={{ backgroundColor: '#FFFFFF', padding: '14px', borderRadius: '6px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '10px' }}>
                Verified Disk Artifacts
              </div>
              <div style={{ fontSize: '11px', fontFamily: 'monospace', color: '#334155', lineHeight: '1.6' }}>
                {STANDARD_ARTIFACTS.map((name) => {
                  const hasIt = dataset.available_artifacts.includes(name);
                  return (
                    <div key={name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span>{name}</span>
                      <span style={{ color: hasIt ? '#16A34A' : '#DC2626', fontWeight: 600 }}>
                        {hasIt ? '✓ Available' : '✗ Missing'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Unified 12-Stage Execution Matrix Table */}
          {stageEntries.length > 0 && (
            <div style={{ backgroundColor: '#FFFFFF', padding: '14px', borderRadius: '6px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '10px' }}>
                Unified 12-Stage Pipeline Execution Matrix
              </div>
              <div className="stages-table-container">
                <table className="data-table" style={{ width: '100%', fontSize: '12px' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left', padding: '8px' }}>Stage Name</th>
                      <th style={{ textAlign: 'left', padding: '8px' }}>Execution Status</th>
                      <th style={{ textAlign: 'left', padding: '8px' }}>Operational Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageEntries.map(([stageName, stageStatus]) => {
                      const sStr = String(stageStatus);
                      let badgeColor = '#64748B';
                      let bgColor = '#F1F5F9';
                      let detailText = 'Evaluated';

                      if (sStr === 'success' || sStr === 'available') {
                        badgeColor = '#16A34A';
                        bgColor = '#DCFCE7';
                        detailText = 'Completed successfully with valid output artifact';
                      } else if (sStr === 'skipped') {
                        badgeColor = '#475569';
                        bgColor = '#F1F5F9';
                        detailText = stageName.includes('youtube')
                          ? 'Omitted during execution (youtube_api_key_not_configured)'
                          : 'Stage execution omitted by configuration or inactive credentials';
                      } else if (sStr === 'partial') {
                        badgeColor = '#D97706';
                        bgColor = '#FEF3C7';
                        detailText = 'Stage executed with fallback or partial platform coverage';
                      } else if (sStr === 'failed') {
                        badgeColor = '#DC2626';
                        bgColor = '#FEE2E2';
                        detailText = 'Encountered execution error during processing';
                      }

                      return (
                        <tr key={stageName}>
                          <td className="font-mono" style={{ padding: '6px 8px' }}>{stageName}</td>
                          <td style={{ padding: '6px 8px' }}>
                            <span
                              style={{
                                display: 'inline-block',
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontSize: '11px',
                                fontWeight: 600,
                                color: badgeColor,
                                backgroundColor: bgColor,
                              }}
                            >
                              {sStr}
                            </span>
                          </td>
                          <td style={{ padding: '6px 8px', color: '#64748B' }}>{detailText}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

