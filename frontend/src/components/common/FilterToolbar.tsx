import React from 'react';
import type { PlatformFilter } from '../../utils/analytics';

interface FilterToolbarProps {
  currentFilter: PlatformFilter;
  onFilterChange: (filter: PlatformFilter) => void;
  filteredCount: number;
  totalCount: number;
  redditCount?: number;
  youtubeCount?: number;
}

export const FilterToolbar: React.FC<FilterToolbarProps> = ({
  currentFilter,
  onFilterChange,
  filteredCount,
  totalCount,
  redditCount,
  youtubeCount,
}) => {
  const redditLabel = typeof redditCount === 'number' ? `Reddit (${redditCount})` : 'Reddit Only';
  const ytLabel =
    typeof youtubeCount === 'number'
      ? youtubeCount === 0
        ? 'YouTube (0 - Skipped)'
        : `YouTube (${youtubeCount})`
      : 'YouTube Only';

  return (
    <div className="filter-toolbar" style={{ position: 'sticky', top: '0', zIndex: 10, backgroundColor: '#FFFFFF', boxShadow: '0 2px 4px rgba(0,0,0,0.04)', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px' }}>
      <div className="filter-buttons-group">
        <span className="filter-label" style={{ fontWeight: 600, color: '#334155' }}>Analytical Scope:</span>
        <button
          type="button"
          className={`filter-btn ${currentFilter === 'all' ? 'active' : ''}`}
          onClick={() => onFilterChange('all')}
        >
          All Platforms ({totalCount})
        </button>
        <button
          type="button"
          className={`filter-btn ${currentFilter === 'reddit' ? 'active' : ''}`}
          onClick={() => onFilterChange('reddit')}
        >
          {redditLabel}
        </button>
        <button
          type="button"
          className={`filter-btn ${currentFilter === 'youtube' ? 'active' : ''}`}
          onClick={() => onFilterChange('youtube')}
        >
          {ytLabel}
        </button>
      </div>

      <div className="filter-status-indicator">
        Scope: <strong>{filteredCount}</strong> of <strong>{totalCount}</strong> records
        {currentFilter !== 'all' && (
          <span className="filter-tag" style={{ marginLeft: '6px' }}>({currentFilter})</span>
        )}
      </div>
    </div>
  );
};

