/**
 * Read-only HTTP client communicating exclusively with the local bridge server (GET only).
 */

import type {
  DatasetListResponse,
  DatasetSummary,
  BridgeApiError,
} from '../types/dataset';
import {
  validateDatasetListResponse,
  validateDatasetSummary,
} from '../validation/validators';

const DEFAULT_BRIDGE_URL = 'http://127.0.0.1:8765';

export function getBridgeBaseUrl(): string {
  const envUrl = import.meta.env.VITE_BRIDGE_URL;
  if (typeof envUrl === 'string' && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return DEFAULT_BRIDGE_URL;
}

export class BridgeClientError extends Error {
  public readonly status: number;
  public readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'BridgeClientError';
    this.status = status;
    this.code = code;
  }
}

async function performGet(path: string, signal?: AbortSignal): Promise<unknown> {
  const baseUrl = getBridgeBaseUrl();
  const url = `${baseUrl}${path.startsWith('/') ? path : `/${path}`}`;

  let resp: Response;
  try {
    resp = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw err;
    }
    const reason = err instanceof Error ? err.message : String(err);
    throw new BridgeClientError(
      0,
      'network_unreachable',
      `Cannot connect to local bridge at ${baseUrl}. Ensure bridge server is running (${reason}).`
    );
  }

  if (!resp.ok) {
    let errorObj: BridgeApiError;
    try {
      const json = await resp.json();
      errorObj = {
        error: json.error || 'server_error',
        message: json.message || `Request failed with HTTP status ${resp.status}`,
        status: resp.status,
      };
    } catch {
      errorObj = {
        error: 'http_error',
        message: `HTTP ${resp.status} ${resp.statusText}`,
        status: resp.status,
      };
    }
    throw new BridgeClientError(resp.status, errorObj.error, errorObj.message);
  }

  try {
    return await resp.json();
  } catch (err) {
    throw new BridgeClientError(
      resp.status,
      'invalid_json',
      `Bridge returned non-JSON response: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

/**
 * Discovers and lists all registered modern runs and legacy baseline.
 */
export async function fetchDatasets(signal?: AbortSignal): Promise<DatasetListResponse> {
  const raw = await performGet('/api/v1/datasets', signal);
  return validateDatasetListResponse(raw);
}

/**
 * Retrieves summary metadata for a specific dataset ID.
 */
export async function fetchDatasetDetails(
  datasetId: string,
  signal?: AbortSignal
): Promise<DatasetSummary> {
  const raw = await performGet(`/api/v1/datasets/${encodeURIComponent(datasetId)}`, signal);
  return validateDatasetSummary(raw);
}

/**
 * Retrieves an authorized analytical artifact for a specific dataset ID.
 */
export async function fetchArtifact(
  datasetId: string,
  artifactName: string,
  signal?: AbortSignal
): Promise<unknown> {
  return performGet(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/artifacts/${encodeURIComponent(artifactName)}`,
    signal
  );
}

