/**
 * Pure analytical calculation engine for Social Media Intelligence Platform.
 * Enforces strict metric traceability, defensive null/invalid handling,
 * and decoupling from presentation components.
 *
 * Authoritative contracts:
 * - src/pipeline/consolidator.py (Schema-v2 record fields & envelope metrics)
 * - src/sentiment/sentiment_pipeline.py
 * - src/misinformation/misinformation_pipeline.py
 * - src/graph/graph_pipeline.py
 * - src/topics/topic_pipeline.py
 */

import type {
  ModernCanonicalRecord,
  ValidatedModernConsolidated,
  ValidatedLegacyConsolidated,
  GraphAnalysisArtifact,
  TopicModelingArtifact,
} from '../types/dataset.ts';

export type PlatformFilter = 'all' | 'reddit' | 'youtube';

// Filter Helpers

/**
 * Filters modern canonical records by selected platform without mutating source data.
 */
export function filterCanonicalRecords(
  records: ModernCanonicalRecord[],
  filter: PlatformFilter
): ModernCanonicalRecord[] {
  if (!Array.isArray(records)) return [];
  if (filter === 'all') return [...records];
  return records.filter((r) => r && r.platform === filter);
}

// Dashboard 1: Executive Overview Analytics

export interface OverviewMetricsResult {
  // Envelope-level metrics (fixed whole-run scope)
  totalRecordsConsolidated: number | null;
  redditCount: number;
  youtubeCount: number;
  youtubeStatusDetail: string | null;
  consolidationStatus: string;
  completeRecordsRun: number;
  degradedRecordsRun: number;
  partialRecordsRun: number;
  unprocessedRecordsRun: number;

  // Record-level metrics (recalculated on active filter)
  filteredRecordsCount: number;
  filteredCompleteCount: number;
  filteredCompletePct: number | null;
  filteredSourcesJoined: Record<string, number>;
  filteredPreprocessingStatus: Record<string, number>;
}

export function calculateOverviewMetrics(
  consolidated: ValidatedModernConsolidated | ValidatedLegacyConsolidated | null,
  filteredRecords: ModernCanonicalRecord[],
  stages?: Record<string, unknown> | null,
  manifest?: Record<string, unknown> | null
): OverviewMetricsResult {
  const isModern =
    consolidated !== null &&
    'schema_version' in consolidated &&
    consolidated.schema_version === '2.0';

  const modernPayload = isModern
    ? (consolidated as ValidatedModernConsolidated)
    : null;

  // Envelope Metrics (Fixed Whole-Run Scope)
  const totalRecordsConsolidated = modernPayload
    ? modernPayload.metrics.total_records_consolidated
    : consolidated?.records?.length ?? null;

  const redditCount = modernPayload?.metrics?.platform_counts?.reddit ?? 0;
  const youtubeCount = modernPayload?.metrics?.platform_counts?.youtube ?? 0;

  // Detect YouTube ingestion stage detail from manifest/stages without guessing
  let youtubeStatusDetail: string | null = null;
  if (youtubeCount === 0) {
    const rawStage =
      (manifest?.stages as Record<string, unknown> | undefined)?.youtube_ingestion ??
      stages?.youtube_ingestion;

    if (rawStage && typeof rawStage === 'object') {
      const stageObj = rawStage as Record<string, unknown>;
      const reason = typeof stageObj.reason === 'string' ? stageObj.reason.trim() : '';
      const status = typeof stageObj.status === 'string' ? stageObj.status.trim() : '';
      if (reason) {
        youtubeStatusDetail = `${status ? `[${status}] ` : ''}${reason}`;
      } else if (status) {
        youtubeStatusDetail = status;
      }
    } else if (typeof rawStage === 'string' && rawStage.trim()) {
      youtubeStatusDetail = rawStage.trim();
    }
  }

  const consolidationStatus = modernPayload?.consolidation_status ?? 'unknown';
  const completeRecordsRun = modernPayload?.metrics?.complete_records ?? 0;
  const degradedRecordsRun = modernPayload?.metrics?.degraded_records ?? 0;
  const partialRecordsRun = modernPayload?.metrics?.partial_records ?? 0;
  const unprocessedRecordsRun = modernPayload?.metrics?.unprocessed_records ?? 0;

  // Record-level metrics (Dynamically Recalculated)
  const filteredRecordsCount = filteredRecords.length;
  let filteredCompleteCount = 0;
  const filteredSourcesJoined: Record<string, number> = {};
  const filteredPreprocessingStatus: Record<string, number> = {};

  for (const rec of filteredRecords) {
    if (rec.consolidation?.is_complete === true) {
      filteredCompleteCount++;
    }
    if (Array.isArray(rec.consolidation?.sources_joined)) {
      for (const src of rec.consolidation.sources_joined) {
        filteredSourcesJoined[src] = (filteredSourcesJoined[src] || 0) + 1;
      }
    }
    const prep = rec.content?.preprocessing_status || 'unknown';
    filteredPreprocessingStatus[prep] = (filteredPreprocessingStatus[prep] || 0) + 1;
  }

  const filteredCompletePct =
    filteredRecordsCount > 0
      ? Math.round((filteredCompleteCount / filteredRecordsCount) * 1000) / 10
      : null;

  return {
    totalRecordsConsolidated,
    redditCount,
    youtubeCount,
    youtubeStatusDetail,
    consolidationStatus,
    completeRecordsRun,
    degradedRecordsRun,
    partialRecordsRun,
    unprocessedRecordsRun,
    filteredRecordsCount,
    filteredCompleteCount,
    filteredCompletePct,
    filteredSourcesJoined,
    filteredPreprocessingStatus,
  };
}

// Dashboard 2: Sentiment Analysis Analytics

export interface SentimentDistribution {
  positive: number;
  neutral: number;
  negative: number;
  positivePct: number | null;
  neutralPct: number | null;
  negativePct: number | null;
}

export interface SentimentMetricsResult {
  // Counts (strictly separated)
  totalRecordsEvaluated: number;
  assessedCount: number;
  undeterminedCount: number;
  unassessedCount: number;

  // Distribution over assessed records only
  distribution: SentimentDistribution;

  // Net sentiment (null if 0 assessed records; never 0.0 by default)
  netSentiment: number | null;

  // Histogram bins over finite scores in [-1.0, 1.0] for assessed records
  scoreHistogram: {
    bins: string[];
    counts: number[];
  };

  // Platform comparison (Reddit vs. YouTube)
  platformBreakdown: {
    reddit: { assessed: number; netSentiment: number | null; positivePct: number | null };
    youtube: { assessed: number; netSentiment: number | null; positivePct: number | null };
  };
}

export function calculateSentimentMetrics(
  records: ModernCanonicalRecord[]
): SentimentMetricsResult {
  const total = Array.isArray(records) ? records.length : 0;
  let positive = 0;
  let neutral = 0;
  let negative = 0;
  let undeterminedCount = 0;
  let unassessedCount = 0;

  const validScores: number[] = [];
  const redditAssessedScores: number[] = [];
  let redditPos = 0;
  let redditNeu = 0;
  let redditNeg = 0;

  const youtubeAssessedScores: number[] = [];
  let youtubePos = 0;
  let youtubeNeu = 0;
  let youtubeNeg = 0;

  for (const rec of records) {
    if (!rec || typeof rec !== 'object') {
      unassessedCount++;
      continue;
    }

    const sent = rec.sentiment;
    if (!sent || typeof sent !== 'object') {
      unassessedCount++;
      continue;
    }

    if (sent.status !== 'success') {
      unassessedCount++;
      continue;
    }

    const rawLabel = (sent as unknown as Record<string, unknown>).label;
    if (rawLabel === 'undetermined') {
      undeterminedCount++;
      continue;
    }

    if (rawLabel === 'positive' || rawLabel === 'neutral' || rawLabel === 'negative') {
      const label = rawLabel;
      if (label === 'positive') positive++;
      else if (label === 'neutral') neutral++;
      else if (label === 'negative') negative++;

      if (
        typeof sent.score === 'number' &&
        Number.isFinite(sent.score) &&
        sent.score >= -1.0 &&
        sent.score <= 1.0
      ) {
        validScores.push(sent.score);
      }

      if (rec.platform === 'reddit') {
        if (label === 'positive') redditPos++;
        else if (label === 'neutral') redditNeu++;
        else if (label === 'negative') redditNeg++;
        if (typeof sent.score === 'number' && Number.isFinite(sent.score)) {
          redditAssessedScores.push(sent.score);
        }
      } else if (rec.platform === 'youtube') {
        if (label === 'positive') youtubePos++;
        else if (label === 'neutral') youtubeNeu++;
        else if (label === 'negative') youtubeNeg++;
        if (typeof sent.score === 'number' && Number.isFinite(sent.score)) {
          youtubeAssessedScores.push(sent.score);
        }
      }
    } else {
      // Unexpected label treated as unassessed/invalid
      unassessedCount++;
    }
  }

  const assessedCount = positive + neutral + negative;

  // Net Sentiment Calculation: only when assessedCount > 0
  let netSentiment: number | null = null;
  let positivePct: number | null = null;
  let neutralPct: number | null = null;
  let negativePct: number | null = null;

  if (assessedCount > 0) {
    netSentiment = Math.round(((positive - negative) / assessedCount) * 1000) / 1000;
    positivePct = Math.round((positive / assessedCount) * 1000) / 10;
    neutralPct = Math.round((neutral / assessedCount) * 1000) / 10;
    negativePct = Math.round((negative / assessedCount) * 1000) / 10;
  }

  // 5 Score Histogram Bins: [-1.0, -0.6), [-0.6, -0.2), [-0.2, 0.2], (0.2, 0.6], (0.6, 1.0]
  const bins = ['[-1.0, -0.6)', '[-0.6, -0.2)', '[-0.2, 0.2]', '(0.2, 0.6]', '(0.6, 1.0]'];
  const counts = [0, 0, 0, 0, 0];
  for (const s of validScores) {
    if (s < -0.6) counts[0]++;
    else if (s < -0.2) counts[1]++;
    else if (s <= 0.2) counts[2]++;
    else if (s <= 0.6) counts[3]++;
    else counts[4]++;
  }

  // Platform comparisons
  const redditAssessed = redditPos + redditNeu + redditNeg;
  const redditNet =
    redditAssessed > 0
      ? Math.round(((redditPos - redditNeg) / redditAssessed) * 1000) / 1000
      : null;
  const redditPosPct =
    redditAssessed > 0
      ? Math.round((redditPos / redditAssessed) * 1000) / 10
      : null;

  const youtubeAssessed = youtubePos + youtubeNeu + youtubeNeg;
  const youtubeNet =
    youtubeAssessed > 0
      ? Math.round(((youtubePos - youtubeNeg) / youtubeAssessed) * 1000) / 1000
      : null;
  const youtubePosPct =
    youtubeAssessed > 0
      ? Math.round((youtubePos / youtubeAssessed) * 1000) / 10
      : null;

  return {
    totalRecordsEvaluated: total,
    assessedCount,
    undeterminedCount,
    unassessedCount,
    distribution: {
      positive,
      neutral,
      negative,
      positivePct,
      neutralPct,
      negativePct,
    },
    netSentiment,
    scoreHistogram: {
      bins,
      counts,
    },
    platformBreakdown: {
      reddit: { assessed: redditAssessed, netSentiment: redditNet, positivePct: redditPosPct },
      youtube: { assessed: youtubeAssessed, netSentiment: youtubeNet, positivePct: youtubePosPct },
    },
  };
}

// Dashboard 3: Topic Intelligence Analytics

export interface TopicItemView {
  topicId: number;
  name: string;
  count: number;
  keywords: { word: string; weight: number | null }[];
}

export interface TopicMetricsResult {
  isAvailable: boolean;
  unavailableReason: string | null;
  totalDocumentsModeled: number | null;
  outlierCount: number;
  topics: TopicItemView[];
}

export const GENERIC_TOPIC_UNAVAILABLE_MESSAGE =
  'Topic artifact unavailable for this dataset run';

export function getTopicUnavailableMessage(
  topicArtifact: unknown,
  stages?: Record<string, unknown> | null,
  manifest?: Record<string, unknown> | null
): string {
  if (topicArtifact !== null && topicArtifact !== undefined) {
    return '';
  }

  // Defensively inspect manifest?.stages?.topics or stages?.topics
  const rawStage =
    (manifest?.stages as Record<string, unknown> | undefined)?.topics ??
    stages?.topics;

  if (rawStage && typeof rawStage === 'object' && !Array.isArray(rawStage)) {
    const stageObj = rawStage as Record<string, unknown>;
    const reason = stageObj.reason;
    if (typeof reason === 'string' && reason.trim().length > 0) {
      return reason;
    }

    const error_category = stageObj.error_category;
    if (typeof error_category === 'string' && error_category.trim().length > 0) {
      return error_category;
    }
  }

  return GENERIC_TOPIC_UNAVAILABLE_MESSAGE;
}

export function extractTopicMetrics(
  topicArtifact: TopicModelingArtifact | null,
  records: ModernCanonicalRecord[],
  stages?: Record<string, unknown> | null,
  manifest?: Record<string, unknown> | null
): TopicMetricsResult {
  if (!topicArtifact || typeof topicArtifact !== 'object') {
    return {
      isAvailable: false,
      unavailableReason: getTopicUnavailableMessage(topicArtifact, stages, manifest),
      totalDocumentsModeled: null,
      outlierCount: 0,
      topics: [],
    };
  }

  const rawInfo = Array.isArray(topicArtifact.topic_info) ? topicArtifact.topic_info : [];
  const topics: TopicItemView[] = [];

  for (const item of rawInfo) {
    if (!item || typeof item !== 'object') continue;
    const tid = typeof item.topic_id === 'number' ? item.topic_id : -1;
    if (tid < 0) continue; // Exclude outlier entry from cluster list

    const rawKeywords = Array.isArray(item.keywords) ? item.keywords : [];
    const keywords: { word: string; weight: number | null }[] = [];
    for (const kw of rawKeywords) {
      if (typeof kw === 'string') {
        keywords.push({ word: kw, weight: null });
      } else if (kw && typeof kw === 'object') {
        const kwObj = kw as Record<string, unknown>;
        const wStr = typeof kwObj.word === 'string' ? kwObj.word : String(kwObj.name || '');
        const wt = typeof kwObj.weight === 'number' ? kwObj.weight : null;
        if (wStr) keywords.push({ word: wStr, weight: wt });
      }
    }

    topics.push({
      topicId: tid,
      name: typeof item.name === 'string' ? item.name : `Topic ${tid}`,
      count: typeof item.count === 'number' ? item.count : 0,
      keywords,
    });
  }

  // Count outliers from records or artifact
  let outlierCount = typeof topicArtifact.outlier_count === 'number' ? topicArtifact.outlier_count : 0;
  if (outlierCount === 0 && Array.isArray(records)) {
    outlierCount = records.filter(
      (r) => r.topics?.status === 'outlier' || r.topics?.topic_id === -1
    ).length;
  }

  return {
    isAvailable: true,
    unavailableReason: null,
    totalDocumentsModeled:
      typeof topicArtifact.total_documents_modeled === 'number'
        ? topicArtifact.total_documents_modeled
        : null,
    outlierCount,
    topics,
  };
}

export interface PlatformTopicMetricsResult {
  isAvailable: boolean;
  emptyStateReason: 'empty_platform' | 'unassessed' | 'none' | null;
  totalDocumentsModeled: number;
  outlierCount: number;
  unassignedCount: number;
  topics: TopicItemView[];
  executionModeDisclosure: string;
}

/**
 * Derives topic clusters dynamically from individual canonical records in the active filter.
 * Ensures that clusters only appear if they contain documents from the active platform.
 */
export function derivePlatformTopicMetrics(
  records: ModernCanonicalRecord[],
  wholeRunTopicArtifact?: TopicModelingArtifact | null,
  platformFilter: PlatformFilter = 'all'
): PlatformTopicMetricsResult {
  if (!Array.isArray(records) || records.length === 0) {
    return {
      isAvailable: false,
      emptyStateReason: 'empty_platform',
      totalDocumentsModeled: 0,
      outlierCount: 0,
      unassignedCount: 0,
      topics: [],
      executionModeDisclosure:
        platformFilter === 'youtube'
          ? 'No YouTube records were ingested in this run (youtube_api_key_not_configured).'
          : 'No records available in the active selection.',
    };
  }

  let unassignedCount = 0;
  let outlierCount = 0;
  const clusterMap = new Map<number, ModernCanonicalRecord[]>();

  for (const r of records) {
    if (
      !r ||
      !r.topics ||
      r.topics.status === 'unavailable' ||
      r.topics.status === 'corrupted' ||
      r.topics.status === 'empty_text' ||
      r.topics.status === 'missing_status'
    ) {
      unassignedCount++;
      continue;
    }

    const tid = typeof r.topics.topic_id === 'number' ? r.topics.topic_id : -1;
    if (tid < 0 || r.topics.status === 'outlier') {
      outlierCount++;
      continue;
    }

    if (!clusterMap.has(tid)) {
      clusterMap.set(tid, []);
    }
    clusterMap.get(tid)!.push(r);
  }

  // Lookup whole-run topic names if artifact is provided
  const artifactTopicNames = new Map<number, string>();
  if (wholeRunTopicArtifact && Array.isArray(wholeRunTopicArtifact.topic_info)) {
    for (const info of wholeRunTopicArtifact.topic_info) {
      if (info && typeof info.topic_id === 'number' && typeof info.name === 'string') {
        artifactTopicNames.set(info.topic_id, info.name);
      }
    }
  }

  const topics: TopicItemView[] = [];

  for (const [tid, clusterRecords] of clusterMap.entries()) {
    const docCount = clusterRecords.length;
    if (docCount === 0) continue;

    // Aggregate keywords from member records
    const wordScores = new Map<string, { sum: number; count: number }>();

    for (const rec of clusterRecords) {
      const kws = Array.isArray(rec.topics?.keywords) ? rec.topics.keywords : [];
      for (const kw of kws) {
        if (!kw || typeof kw.word !== 'string') continue;
        const w = kw.word.trim();
        if (!w) continue;
        const wt = typeof kw.weight === 'number' && Number.isFinite(kw.weight) ? kw.weight : 0.1;
        const current = wordScores.get(w) || { sum: 0, count: 0 };
        current.sum += wt;
        current.count += 1;
        wordScores.set(w, current);
      }
    }

    // Sort keywords by average weight descending
    const sortedKeywords = Array.from(wordScores.entries())
      .map(([word, stat]) => ({
        word,
        weight: Math.round((stat.sum / stat.count) * 10000) / 10000,
      }))
      .sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0))
      .slice(0, 5);

    const name =
      artifactTopicNames.get(tid) ||
      (sortedKeywords.length > 0 ? `${tid}_${sortedKeywords[0].word}` : `Topic #${tid}`);

    topics.push({
      topicId: tid,
      name,
      count: docCount,
      keywords: sortedKeywords,
    });
  }

  topics.sort((a, b) => b.count - a.count || a.topicId - b.topicId);

  const totalModeled = records.length - unassignedCount;

  let executionModeDisclosure = 'Topic clusters derived from individual record assignments.';
  if (wholeRunTopicArtifact?.execution_mode === 'statistical_fallback') {
    executionModeDisclosure =
      'Topic modeling executed via statistical fallback (c-TF-IDF / MiniBatchKMeans) due to small corpus size (< 15 documents). Cluster probabilities reflect heuristic weights, not calibrated Bayesian posteriors.';
  } else if (wholeRunTopicArtifact?.execution_mode === 'neural_bertopic') {
    executionModeDisclosure =
      'Topic clusters discovered via neural BERTopic with multilingual sentence embeddings and HDBSCAN density clustering.';
  }

  return {
    isAvailable: true,
    emptyStateReason: totalModeled === 0 ? 'unassessed' : null,
    totalDocumentsModeled: totalModeled,
    outlierCount,
    unassignedCount,
    topics,
    executionModeDisclosure,
  };
}

// Empty State Infrastructure

export interface PlatformEmptyStateInfo {
  isEmpty: boolean;
  platform: PlatformFilter;
  title: string;
  reason: string;
  manifestReason?: string;
  actionHint: string;
}

export function getPlatformEmptyState(
  platform: PlatformFilter,
  recordCount: number,
  stages?: Record<string, unknown> | null
): PlatformEmptyStateInfo {
  if (recordCount > 0) {
    return {
      isEmpty: false,
      platform,
      title: '',
      reason: '',
      actionHint: '',
    };
  }

  if (platform === 'youtube') {
    const ytStage = stages?.youtube_ingestion as Record<string, unknown> | undefined;
    const manifestReason =
      typeof ytStage?.reason === 'string' ? ytStage.reason : 'youtube_api_key_not_configured';

    return {
      isEmpty: true,
      platform: 'youtube',
      title: 'No YouTube Records in Active Execution',
      reason:
        'YouTube ingestion was skipped during pipeline execution because the YouTube Data API key was not configured.',
      manifestReason,
      actionHint: "Switch the platform filter above to 'All Platforms' or 'Reddit Only' to explore active analytical results.",
    };
  }

  if (platform === 'reddit') {
    return {
      isEmpty: true,
      platform: 'reddit',
      title: 'No Reddit Records in Active Execution',
      reason: 'Reddit ingestion yielded zero records in this pipeline run.',
      actionHint: "Switch the platform filter above to 'All Platforms' or select a different dataset.",
    };
  }

  return {
    isEmpty: true,
    platform: 'all',
    title: 'No Records Available in This Dataset',
    reason: 'The selected dataset contains zero canonical records.',
    actionHint: 'Select another execution run from the dataset dropdown above.',
  };
}

// Dashboard 4: Misinformation Risk Analytics
// Note: veracity_label is strictly excluded from all outputs!

export interface MisinformationMetricsResult {
  totalRecordsEvaluated: number;
  assessedCount: number;
  unassessedCount: number;

  // Heuristic Credibility Index (0.0 to 1.0)
  averageCredibilityScore: number | null;

  // Risk Tier counts (low, moderate, high_risk) for assessed records
  riskTiers: {
    low: number;
    moderate: number;
    highRisk: number;
  };

  // Histogram over valid credibility scores [0.0, 1.0]
  scoreHistogram: {
    bins: string[];
    counts: number[];
  };

  // Aggregated genuine discourse flags from successfully assessed records
  detectedSignals: { flag: string; count: number }[];

  // Available stylometric feature averages (independent denominators)
  stylometricAverages: Record<string, number>;

  epistemicDisclosure: string;
}

export const METHODOLOGICAL_DISCLOSURE_TEXT =
  'Detected patterns represent lexical and stylistic markers, not verified factual misinformation. Credibility scores are heuristic model estimates, not calibrated probabilities.';

export function calculateMisinformationMetrics(
  records: ModernCanonicalRecord[]
): MisinformationMetricsResult {
  const total = Array.isArray(records) ? records.length : 0;
  let assessedCount = 0;
  let unassessedCount = 0;

  const validCredScores: number[] = [];
  let lowCount = 0;
  let modCount = 0;
  let highCount = 0;

  const signalCounts: Record<string, number> = {};
  const styleSums: Record<string, { sum: number; count: number }> = {};

  for (const rec of records) {
    if (!rec || typeof rec !== 'object') {
      unassessedCount++;
      continue;
    }

    const mis = rec.misinformation;
    if (!mis || typeof mis !== 'object') {
      unassessedCount++;
      continue;
    }

    if (mis.status !== 'success') {
      unassessedCount++;
      continue;
    }

    assessedCount++;

    // Credibility score handling (must be finite number in [0.0, 1.0])
    // Per Correction 2: Records with missing or invalid scores are strictly excluded from risk calculations.
    // Stored risk_tier does not silently compensate for an absent/invalid score.
    if (
      typeof mis.credibility_score === 'number' &&
      Number.isFinite(mis.credibility_score) &&
      mis.credibility_score >= 0.0 &&
      mis.credibility_score <= 1.0
    ) {
      const score = mis.credibility_score;
      validCredScores.push(score);

      // Backend canonical risk thresholds: < 0.40 high, 0.40..0.70 mod, >= 0.70 low
      if (score < 0.4) {
        highCount++;
      } else if (score < 0.7) {
        modCount++;
      } else {
        lowCount++;
      }
    }

    // Signals detected (aggregate ONLY actual non-empty strings from successful records)
    const rawSignals = (rec as unknown as Record<string, unknown>).signals_detected ??
      (mis as unknown as Record<string, unknown>).signals_detected;
    if (Array.isArray(rawSignals)) {
      for (const sig of rawSignals) {
        if (typeof sig === 'string' && sig.trim()) {
          const cleanSig = sig.trim();
          signalCounts[cleanSig] = (signalCounts[cleanSig] || 0) + 1;
        }
      }
    }

    // Stylometric signals (independent denominators for numeric values)
    if (mis.stylometric_signals && typeof mis.stylometric_signals === 'object') {
      for (const [key, val] of Object.entries(mis.stylometric_signals)) {
        if (typeof val === 'number' && Number.isFinite(val)) {
          if (!styleSums[key]) {
            styleSums[key] = { sum: 0, count: 0 };
          }
          styleSums[key].sum += val;
          styleSums[key].count += 1;
        }
      }
    }
  }

  // Calculate average credibility score
  const averageCredibilityScore =
    validCredScores.length > 0
      ? Math.round(
          (validCredScores.reduce((acc, curr) => acc + curr, 0) /
            validCredScores.length) *
            1000
        ) / 1000
      : null;

  // 4 Score Histogram Bins: [0.0, 0.40), [0.40, 0.70), [0.70, 0.85), [0.85, 1.0]
  const bins = ['[0.0, 0.40) High Risk', '[0.40, 0.70) Moderate', '[0.70, 0.85) Low Risk', '[0.85, 1.0] Verified Baseline'];
  const counts = [0, 0, 0, 0];
  for (const s of validCredScores) {
    if (s < 0.4) counts[0]++;
    else if (s < 0.7) counts[1]++;
    else if (s < 0.85) counts[2]++;
    else counts[3]++;
  }

  // Convert signals to sorted list
  const detectedSignals = Object.entries(signalCounts)
    .map(([flag, count]) => ({ flag, count }))
    .sort((a, b) => b.count - a.count);

  // Stylometric averages
  const stylometricAverages: Record<string, number> = {};
  for (const [k, v] of Object.entries(styleSums)) {
    if (v.count > 0) {
      stylometricAverages[k] = Math.round((v.sum / v.count) * 10000) / 10000;
    }
  }

  return {
    totalRecordsEvaluated: total,
    assessedCount,
    unassessedCount,
    averageCredibilityScore,
    riskTiers: {
      low: lowCount,
      moderate: modCount,
      highRisk: highCount,
    },
    scoreHistogram: {
      bins,
      counts,
    },
    detectedSignals,
    stylometricAverages,
    epistemicDisclosure: METHODOLOGICAL_DISCLOSURE_TEXT,
  };
}

// Dashboard 5: Network & Influencer Intelligence Analytics

export interface NetworkNodeItem {
  nodeId: string;
  platform: string;
  identityType: string;
  username: string;
  activityCount: number;
  inDegree?: number;
  outDegree?: number;
}

export interface NetworkEdgeItem {
  source: string;
  target: string;
  weight: number;
}

export interface NetworkInfluencerItem {
  nodeId: string;
  username: string;
  platform: string;
  influencerScore: number | null;
  pagerank: number | null;
  inDegree: number;
  outDegree: number;
}

export interface NetworkMetricsResult {
  isAvailable: boolean;
  totalUsers: number;
  totalInteractions: number;
  density: number;
  communitiesCount: number;
  isDirected: boolean;
  nodes: NetworkNodeItem[];
  edges?: NetworkEdgeItem[];
  influencers: NetworkInfluencerItem[];
}

export function extractNetworkMetrics(
  graphArtifact: GraphAnalysisArtifact | null
): NetworkMetricsResult {
  if (!graphArtifact || typeof graphArtifact !== 'object') {
    return {
      isAvailable: false,
      totalUsers: 0,
      totalInteractions: 0,
      density: 0,
      communitiesCount: 0,
      isDirected: false,
      nodes: [],
      influencers: [],
    };
  }

  const metrics = graphArtifact.graph_metrics || {
    total_users: 0,
    total_interactions: 0,
    density: 0,
    communities_count: 0,
    is_directed: true,
  };

  // Derive node metadata following verified schema fields (identity_type, platform, node_id)
  const rawNodes = Array.isArray(graphArtifact.nodes) ? graphArtifact.nodes : [];
  const nodes: NetworkNodeItem[] = [];

  for (const n of rawNodes) {
    if (!n || typeof n !== 'object') continue;
    const nObj = n as Record<string, unknown>;
    nodes.push({
      nodeId: String(nObj.node_id || nObj.id || 'unknown'),
      platform: String(nObj.platform || 'unknown'),
      identityType: String(nObj.identity_type || 'unspecified'),
      username: String(nObj.username || nObj.node_id || ''),
      activityCount: typeof nObj.activity_count === 'number' ? nObj.activity_count : 0,
    });
  }

  // Derive influencers from top_influencers array
  const rawInf = Array.isArray(graphArtifact.top_influencers)
    ? graphArtifact.top_influencers
    : [];
  const influencers: NetworkInfluencerItem[] = [];

  for (const inf of rawInf) {
    if (!inf || typeof inf !== 'object') continue;
    const iObj = inf as Record<string, unknown>;
    influencers.push({
      nodeId: String(iObj.node_id || ''),
      username: String(iObj.username || iObj.node_id || ''),
      platform: String(iObj.platform || 'reddit'),
      influencerScore: typeof iObj.influencer_score === 'number' ? iObj.influencer_score : null,
      pagerank: typeof iObj.pagerank === 'number' ? iObj.pagerank : null,
      inDegree: typeof iObj.in_degree_raw === 'number' ? iObj.in_degree_raw : (typeof iObj.in_degree === 'number' ? iObj.in_degree : 0),
      outDegree: typeof iObj.out_degree_raw === 'number' ? iObj.out_degree_raw : (typeof iObj.out_degree === 'number' ? iObj.out_degree : 0),
    });
  }

  return {
    isAvailable: true,
    totalUsers: metrics.total_users,
    totalInteractions: metrics.total_interactions,
    density: metrics.density,
    communitiesCount: metrics.communities_count,
    isDirected: metrics.is_directed,
    nodes,
    influencers,
  };
}

/**
 * Derives network topology and influencer rankings scoped to the selected platform.
 * Enforces edge-endpoint consistency (removing orphaned edges) and suppresses
 * misleading influencer scores for isolated singleton accounts (degree 0).
 */
export function derivePlatformGraphMetrics(
  graphArtifact: GraphAnalysisArtifact | null,
  platformFilter: PlatformFilter = 'all'
): NetworkMetricsResult {
  if (!graphArtifact || typeof graphArtifact !== 'object') {
    return {
      isAvailable: false,
      totalUsers: 0,
      totalInteractions: 0,
      density: 0,
      communitiesCount: 0,
      isDirected: false,
      nodes: [],
      influencers: [],
    };
  }

  const rawNodes = Array.isArray(graphArtifact.nodes) ? graphArtifact.nodes : [];
  const retainedNodes =
    platformFilter === 'all'
      ? rawNodes
      : rawNodes.filter((n) => {
          if (!n || typeof n !== 'object') return false;
          const p = String((n as Record<string, unknown>).platform || '').toLowerCase();
          return p === platformFilter.toLowerCase();
        });

  const retainedNodeIds = new Set<string>();
  const nodes: NetworkNodeItem[] = [];

  for (const n of retainedNodes) {
    if (!n || typeof n !== 'object') continue;
    const nObj = n as Record<string, unknown>;
    const nodeId = String(nObj.node_id || nObj.id || 'unknown');
    retainedNodeIds.add(nodeId);
    nodes.push({
      nodeId,
      platform: String(nObj.platform || 'unknown'),
      identityType: String(nObj.identity_type || 'unspecified'),
      username: String(nObj.username || nObj.node_id || ''),
      activityCount: typeof nObj.activity_count === 'number' ? nObj.activity_count : 0,
    });
  }

  // Endpoint consistency: both source and target must belong to retainedNodeIds
  const rawEdges = Array.isArray(graphArtifact.edges) ? graphArtifact.edges : [];
  const retainedEdges = rawEdges.filter((e) => {
    if (!e || typeof e !== 'object') return false;
    const eObj = e as Record<string, unknown>;
    const src = String(eObj.source || '');
    const tgt = String(eObj.target || '');
    return retainedNodeIds.has(src) && retainedNodeIds.has(tgt);
  });

  const totalUsers = nodes.length;
  const totalInteractions = retainedEdges.length;

  // Strict directed graph density calculation: |E| / (|V| * (|V| - 1))
  const density =
    totalUsers > 1
      ? Math.round((totalInteractions / (totalUsers * (totalUsers - 1))) * 1000000) / 1000000
      : 0.0;

  // Track in-degree and out-degree from retained edges
  const inDegreeMap = new Map<string, number>();
  const outDegreeMap = new Map<string, number>();

  for (const e of retainedEdges) {
    const eObj = e as Record<string, unknown>;
    const src = String(eObj.source || '');
    const tgt = String(eObj.target || '');
    outDegreeMap.set(src, (outDegreeMap.get(src) || 0) + 1);
    inDegreeMap.set(tgt, (inDegreeMap.get(tgt) || 0) + 1);
  }

  // Derive influencers with sanity check:
  // An account is ONLY an influencer if it participates in at least one active edge (degree > 0).
  // Isolated singletons with 0 edges must NOT be ranked as influencers!
  const influencers: NetworkInfluencerItem[] = [];

  if (totalInteractions > 0) {
    const rawInf = Array.isArray(graphArtifact.top_influencers)
      ? graphArtifact.top_influencers
      : [];

    for (const inf of rawInf) {
      if (!inf || typeof inf !== 'object') continue;
      const iObj = inf as Record<string, unknown>;
      const nodeId = String(iObj.node_id || '');
      if (!retainedNodeIds.has(nodeId)) continue;

      const inDeg = inDegreeMap.get(nodeId) || 0;
      const outDeg = outDegreeMap.get(nodeId) || 0;
      if (inDeg === 0 && outDeg === 0) continue; // Suppress isolated singletons

      influencers.push({
        nodeId,
        username: String(iObj.username || iObj.node_id || ''),
        platform: String(iObj.platform || 'reddit'),
        influencerScore: typeof iObj.influencer_score === 'number' ? iObj.influencer_score : null,
        pagerank: typeof iObj.pagerank === 'number' ? iObj.pagerank : null,
        inDegree: inDeg,
        outDegree: outDeg,
      });
    }
  }

  // Attach accurate in-degree and out-degree to nodes
  const nodesWithDegrees: NetworkNodeItem[] = nodes.map((n) => ({
    ...n,
    inDegree: inDegreeMap.get(n.nodeId) || 0,
    outDegree: outDegreeMap.get(n.nodeId) || 0,
  }));

  const typedEdges: NetworkEdgeItem[] = retainedEdges.map((e) => {
    const eObj = e as Record<string, unknown>;
    return {
      source: String(eObj.source || ''),
      target: String(eObj.target || ''),
      weight: typeof eObj.weight === 'number' ? eObj.weight : 1,
    };
  });

  return {
    isAvailable: totalUsers > 0,
    totalUsers,
    totalInteractions,
    density,
    communitiesCount:
      totalUsers > 0
        ? totalInteractions > 0
          ? (graphArtifact.graph_metrics?.communities_count ?? 1)
          : totalUsers
        : 0,
    isDirected: true,
    nodes: nodesWithDegrees,
    edges: typedEdges,
    influencers,
  };
}

// Feed Explorer Search & Filtering

export interface FeedFilterOptions {
  query?: string;
  sentimentFilter?: 'all' | 'positive' | 'neutral' | 'negative' | 'undetermined';
  topicFilter?: number | 'all' | 'outlier';
  sortBy?: 'default' | 'newest' | 'score' | 'credibility';
}

/**
 * Pure search, filtering, and sorting for the Social Feed Explorer.
 * Matches text case-insensitively across title, body, author, and channel title.
 */
export function filterAndSortFeedRecords(
  records: ModernCanonicalRecord[],
  options: FeedFilterOptions = {}
): ModernCanonicalRecord[] {
  if (!Array.isArray(records)) return [];

  const {
    query = '',
    sentimentFilter = 'all',
    topicFilter = 'all',
    sortBy = 'default',
  } = options;

  let result = records;

  // 1. Text Search across title, body, author username, and channel title
  const q = query.trim().toLowerCase();
  if (q) {
    result = result.filter((r) => {
      if (!r) return false;
      const title = (r.content?.title_cleaned || r.content?.title_raw || '').toLowerCase();
      const body = (r.content?.body_cleaned || r.content?.body_raw || '').toLowerCase();
      const username = (r.author?.username || '').toLowerCase();
      const channel = (r.author?.channel_title || '').toLowerCase();
      return title.includes(q) || body.includes(q) || username.includes(q) || channel.includes(q);
    });
  }

  // 2. Sentiment Filter
  if (sentimentFilter !== 'all') {
    result = result.filter((r) => {
      if (!r) return false;
      if (sentimentFilter === 'undetermined') {
        return r.sentiment?.status !== 'success' || !r.sentiment?.label;
      }
      return r.sentiment?.status === 'success' && r.sentiment?.label === sentimentFilter;
    });
  }

  // 3. Topic Filter
  if (topicFilter !== 'all') {
    result = result.filter((r) => {
      if (!r) return false;
      if (topicFilter === 'outlier') {
        return r.topics?.topic_id === -1;
      }
      return r.topics?.topic_id === topicFilter;
    });
  }

  // 4. Sorting
  if (sortBy === 'newest') {
    result = [...result].sort((a, b) => {
      const ta = a.timestamps?.published_at ? new Date(a.timestamps.published_at).getTime() : 0;
      const tb = b.timestamps?.published_at ? new Date(b.timestamps.published_at).getTime() : 0;
      return tb - ta;
    });
  } else if (sortBy === 'score') {
    result = [...result].sort((a, b) => {
      const sa = typeof a.engagement?.score === 'number' ? a.engagement.score : -1;
      const sb = typeof b.engagement?.score === 'number' ? b.engagement.score : -1;
      return sb - sa;
    });
  } else if (sortBy === 'credibility') {
    result = [...result].sort((a, b) => {
      const ca = typeof a.misinformation?.credibility_score === 'number' ? a.misinformation.credibility_score : -1;
      const cb = typeof b.misinformation?.credibility_score === 'number' ? b.misinformation.credibility_score : -1;
      return cb - ca;
    });
  }

  return result;
}
