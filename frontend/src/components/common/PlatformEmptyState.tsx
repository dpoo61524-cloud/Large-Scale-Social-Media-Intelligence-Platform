import React from 'react';
import type { PlatformEmptyStateInfo, PlatformFilter } from '../../utils/analytics';

interface PlatformEmptyStateProps {
  info: PlatformEmptyStateInfo;
  onResetFilter?: (platform: PlatformFilter) => void;
}

export const PlatformEmptyState: React.FC<PlatformEmptyStateProps> = ({
  info,
  onResetFilter,
}) => {
  return (
    <div className="card-panel platform-empty-state-panel" style={{ marginTop: '24px', padding: '32px 24px', textAlign: 'center' }}>
      <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', borderRadius: '50%', backgroundColor: '#EFF6FF', color: '#2563EB', fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
        ℹ
      </div>
      <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#1E293B', marginBottom: '8px' }}>
        {info.title}
      </h3>
      <p style={{ fontSize: '14px', color: '#64748B', maxWidth: '600px', margin: '0 auto 16px auto', lineHeight: '1.5' }}>
        {info.reason}
      </p>

      {info.manifestReason && (
        <div style={{ display: 'inline-block', backgroundColor: '#F1F5F9', padding: '4px 12px', borderRadius: '4px', fontSize: '12px', fontFamily: 'monospace', color: '#475569', marginBottom: '20px' }}>
          Execution Manifest Reason: <strong>{info.manifestReason}</strong>
        </div>
      )}

      {info.actionHint && (
        <div style={{ fontSize: '13px', color: '#3B82F6', fontWeight: 500 }}>
          {info.actionHint}
        </div>
      )}

      {onResetFilter && (
        <div style={{ marginTop: '20px' }}>
          <button
            type="button"
            className="filter-btn active"
            onClick={() => onResetFilter('all')}
            style={{ cursor: 'pointer', padding: '8px 16px', fontSize: '13px' }}
          >
            Switch to All Platforms
          </button>
        </div>
      )}
    </div>
  );
};

