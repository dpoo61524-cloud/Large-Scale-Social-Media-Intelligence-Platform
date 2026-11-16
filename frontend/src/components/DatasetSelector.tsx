import React from 'react';
import type { DatasetSummary } from '../types/dataset';

interface DatasetSelectorProps {
  datasets: DatasetSummary[];
  selectedId: string | null;
  onSelect: (datasetId: string) => void;
  isLoading: boolean;
  onRefresh: () => void;
}

export const DatasetSelector: React.FC<DatasetSelectorProps> = ({
  datasets,
  selectedId,
  onSelect,
  isLoading,
  onRefresh,
}) => {
  const modernRuns = datasets.filter((d) => d.type === 'modern');
  const legacyDatasets = datasets.filter((d) => d.type === 'legacy');

  return (
    <div className="dataset-selector-card">
      <div className="selector-header">
        <label htmlFor="dataset-select" className="selector-label">
          Execution Dataset
        </label>
        <button
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
          className="refresh-button"
        >
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="selector-control-wrapper">
        <select
          id="dataset-select"
          className="dataset-dropdown"
          value={selectedId || ''}
          onChange={(e) => onSelect(e.target.value)}
          disabled={isLoading || datasets.length === 0}
        >
          {datasets.length === 0 && <option value="">No datasets available</option>}

          {modernRuns.length > 0 && (
            <optgroup label="Intelligence Pipeline Runs">
              {modernRuns.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id} — Status: {d.status} ({d.available_artifacts.length} verified datasets)
                </option>
              ))}
            </optgroup>
          )}

          {legacyDatasets.length > 0 && (
            <optgroup label="Historical Baseline Archive">
              {legacyDatasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label} ({d.available_artifacts.length} verified datasets)
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </div>
    </div>
  );
};

