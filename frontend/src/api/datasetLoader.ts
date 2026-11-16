/**
 * Dataset artifact loading and asynchronous request coordination.
 * Enforces strict dataset isolation, request cancellation, and stale response protection.
 */

import type {
  DatasetSummary,
  ValidatedDatasetData,
  ValidatedModernConsolidated,
  ValidatedLegacyConsolidated,
  SentimentAnalysisArtifact,
  MisinformationAnalysisArtifact,
  GraphAnalysisArtifact,
  TopicModelingArtifact,
} from '../types/dataset.ts';
import {
  validateModernConsolidated,
  validateLegacyConsolidated,
  validateSentimentArtifact,
  validateMisinformationArtifact,
  validateGraphArtifact,
  validateTopicArtifact,
} from '../validation/validators.ts';

export type ArtifactFetcher = (
  datasetId: string,
  artifactName: string,
  signal?: AbortSignal
) => Promise<unknown>;

/**
 * Loads and validates available artifacts for a specific dataset.
 * Enforces dataset isolation: missing modern artifacts NEVER fall back to legacy data.
 */
export async function loadDatasetArtifacts(
  summary: DatasetSummary,
  fetchArtifact: ArtifactFetcher,
  signal?: AbortSignal
): Promise<ValidatedDatasetData> {
  if (signal?.aborted) {
    throw new DOMException('Request aborted', 'AbortError');
  }

  let validatedConsolidated: ValidatedModernConsolidated | ValidatedLegacyConsolidated | null = null;
  let validatedSentiment: SentimentAnalysisArtifact | null = null;
  let validatedMisinfo: MisinformationAnalysisArtifact | null = null;
  let validatedGraph: GraphAnalysisArtifact | null = null;
  let validatedTopic: TopicModelingArtifact | null = null;
  let topicReason: string | undefined = undefined;

  // 1. Consolidated Intelligence (strictly isolated between modern and legacy)
  if (summary.available_artifacts.includes('consolidated_intelligence.json')) {
    try {
      const raw = await fetchArtifact(summary.id, 'consolidated_intelligence.json', signal);
      if (signal?.aborted) {
        throw new DOMException('Request aborted', 'AbortError');
      }
      if (summary.type === 'modern') {
        validatedConsolidated = validateModernConsolidated(raw);
      } else {
        validatedConsolidated = validateLegacyConsolidated(raw);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      // Failed artifact remains null; do NOT fallback to other datasets
      validatedConsolidated = null;
    }
  }

  // 2. Sentiment Analysis
  if (summary.available_artifacts.includes('sentiment_analysis_results.json')) {
    try {
      const raw = await fetchArtifact(summary.id, 'sentiment_analysis_results.json', signal);
      if (signal?.aborted) {
        throw new DOMException('Request aborted', 'AbortError');
      }
      validatedSentiment = validateSentimentArtifact(raw);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      validatedSentiment = null;
    }
  }

  // 3. Misinformation Analysis
  if (summary.available_artifacts.includes('misinformation_analysis_results.json')) {
    try {
      const raw = await fetchArtifact(summary.id, 'misinformation_analysis_results.json', signal);
      if (signal?.aborted) {
        throw new DOMException('Request aborted', 'AbortError');
      }
      validatedMisinfo = validateMisinformationArtifact(raw);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      validatedMisinfo = null;
    }
  }

  // 4. Graph Analysis
  if (summary.available_artifacts.includes('graph_analysis_results.json')) {
    try {
      const raw = await fetchArtifact(summary.id, 'graph_analysis_results.json', signal);
      if (signal?.aborted) {
        throw new DOMException('Request aborted', 'AbortError');
      }
      validatedGraph = validateGraphArtifact(raw);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      validatedGraph = null;
    }
  }

  // 5. Topic Modeling (strictly isolated; never fall back to legacy data!)
  if (summary.available_artifacts.includes('topic_modeling_results.json')) {
    try {
      const raw = await fetchArtifact(summary.id, 'topic_modeling_results.json', signal);
      if (signal?.aborted) {
        throw new DOMException('Request aborted', 'AbortError');
      }
      validatedTopic = validateTopicArtifact(raw);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      validatedTopic = null;
    }
  } else {
    // Explicit unavailable reason
    if (summary.type === 'modern') {
      topicReason =
        'Topic modeling stage failed during pipeline execution; artifact not produced. No legacy fallback permitted.';
    } else {
      topicReason = 'Topic modeling artifact is absent in this dataset.';
    }
  }

  if (signal?.aborted) {
    throw new DOMException('Request aborted', 'AbortError');
  }

  return {
    datasetId: summary.id,
    datasetType: summary.type,
    consolidated: validatedConsolidated,
    sentiment: validatedSentiment,
    misinformation: validatedMisinfo,
    graph: validatedGraph,
    topic: validatedTopic,
    topicUnavailableReason: topicReason,
    retrievedAt: new Date().toISOString(),
  };
}

/**
 * Manages dataset request concurrency, cancellation of in-flight requests,
 * and stale response prevention.
 */
export class DatasetRequestCoordinator {
  private currentAbortController: AbortController | null = null;
  private activeRequestId = 0;

  /**
   * Switches to a new dataset, immediately aborting any ongoing requests.
   * If a previous request resolves after switching, it is discarded.
   */
  public async switchDataset(
    summary: DatasetSummary,
    fetchArtifact: ArtifactFetcher
  ): Promise<ValidatedDatasetData | null> {
    // 1. Abort previous in-flight request
    if (this.currentAbortController) {
      this.currentAbortController.abort();
    }

    // 2. Increment request ID and create new controller
    this.activeRequestId += 1;
    const currentRequestId = this.activeRequestId;

    const controller = new AbortController();
    this.currentAbortController = controller;
    const signal = controller.signal;

    try {
      const data = await loadDatasetArtifacts(summary, fetchArtifact, signal);

      // Stale response guard: discard if a newer request was initiated
      if (this.activeRequestId !== currentRequestId || signal.aborted) {
        return null;
      }

      return data;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return null;
      }
      // If active request changed, swallow error
      if (this.activeRequestId !== currentRequestId) {
        return null;
      }
      throw err;
    }
  }

  public abortActive(): void {
    if (this.currentAbortController) {
      this.currentAbortController.abort();
      this.currentAbortController = null;
    }
  }

  public getActiveRequestId(): number {
    return this.activeRequestId;
  }
}
