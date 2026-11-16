import React from 'react';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const normalized = status.toLowerCase();

  let badgeClass = 'status-badge-neutral';
  if (normalized === 'success') {
    badgeClass = 'status-badge-success';
  } else if (normalized === 'failed' || normalized === 'error') {
    badgeClass = 'status-badge-failed';
  } else if (normalized === 'partial') {
    badgeClass = 'status-badge-partial';
  } else if (normalized === 'skipped' || normalized === 'empty') {
    badgeClass = 'status-badge-muted';
  } else if (normalized === 'legacy' || normalized === 'historical') {
    badgeClass = 'status-badge-legacy';
  }

  return (
    <span className={`status-badge ${badgeClass} ${className}`}>
      {status}
    </span>
  );
};

