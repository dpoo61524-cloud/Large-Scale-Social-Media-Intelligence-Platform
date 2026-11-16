import React from 'react';

export const ModelBenchmarkCard: React.FC = () => {
  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8D0',
        borderRadius: '8px',
        padding: '16px 20px',
        marginBottom: '20px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
      }}
    >
      {/* Title & Badge (No Description) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
          Supervised Classifier Benchmark Audit
        </div>
        <span
          style={{
            fontSize: '11px',
            fontWeight: 700,
            color: '#064E3B',
            backgroundColor: '#FEF9C3',
            padding: '3px 10px',
            borderRadius: '12px',
            border: '1px solid #E2E8D0',
          }}
        >
          Model Status: Validated
        </span>
      </div>

      {/* 4 Clean Metric Cards in a Single Row (Confusion Matrix Deleted) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
        <div style={{ padding: '12px 14px', backgroundColor: '#FEFCE8', borderRadius: '6px', border: '1px solid #E2E8D0' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#4B6358', textTransform: 'uppercase' }}>
            Accuracy
          </div>
          <div style={{ fontSize: '22px', fontWeight: 800, color: '#064E3B', fontFamily: 'monospace', marginTop: '3px' }}>
            70.0%
          </div>
          <div style={{ fontSize: '10.5px', color: '#047857', marginTop: '2px' }}>
            Stratified 3-fold evaluation
          </div>
        </div>

        <div style={{ padding: '12px 14px', backgroundColor: '#FEFCE8', borderRadius: '6px', border: '1px solid #E2E8D0' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#4B6358', textTransform: 'uppercase' }}>
            F1-Score (Macro)
          </div>
          <div style={{ fontSize: '22px', fontWeight: 800, color: '#064E3B', fontFamily: 'monospace', marginTop: '3px' }}>
            68.8%
          </div>
          <div style={{ fontSize: '10.5px', color: '#047857', marginTop: '2px' }}>
            Balanced harmonic mean
          </div>
        </div>

        <div style={{ padding: '12px 14px', backgroundColor: '#FEFCE8', borderRadius: '6px', border: '1px solid #E2E8D0' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#4B6358', textTransform: 'uppercase' }}>
            Precision (Macro)
          </div>
          <div style={{ fontSize: '22px', fontWeight: 800, color: '#064E3B', fontFamily: 'monospace', marginTop: '3px' }}>
            68.8%
          </div>
          <div style={{ fontSize: '10.5px', color: '#047857', marginTop: '2px' }}>
            Low false alarm rate
          </div>
        </div>

        <div style={{ padding: '12px 14px', backgroundColor: '#FEFCE8', borderRadius: '6px', border: '1px solid #E2E8D0' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#4B6358', textTransform: 'uppercase' }}>
            ROC-AUC Metric
          </div>
          <div style={{ fontSize: '22px', fontWeight: 800, color: '#064E3B', fontFamily: 'monospace', marginTop: '3px' }}>
            0.667
          </div>
          <div style={{ fontSize: '10.5px', color: '#047857', marginTop: '2px' }}>
            Discriminative capability
          </div>
        </div>
      </div>
    </div>
  );
};
