/**
 * Comprehensive unit tests for Phase 3.3 analytical calculation engine.
 * Validates calculation rules, defensive null/invalid handling,
 * dataset isolation, filtering recalculations, and metric scope boundaries.
 *
 * Runs under Node.js native test runner: node --experimental-strip-types --test
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  filterCanonicalRecords,
  calculateOverviewMetrics,
  calculateSentimentMetrics,
  getTopicUnavailableMessage,
  extractTopicMetrics,
  derivePlatformTopicMetrics,
  calculateMisinformationMetrics,
  METHODOLOGICAL_DISCLOSURE_TEXT,
  extractNetworkMetrics,
  derivePlatformGraphMetrics,
  getPlatformEmptyState,
  filterAndSortFeedRecords,
} from '../utils/analytics.ts';

import type {
  ModernCanonicalRecord,
  ValidatedModernConsolidated,
  GraphAnalysisArtifact,
  TopicModelingArtifact,
} from '../types/dataset.ts';

// Test Fixtures

function createSampleRecord(
  id: string,
  platform: 'reddit' | 'youtube',
  overrides: Record<string, unknown> = {}
): ModernCanonicalRecord {
  const docId = `${platform}_${id}`;
  const base: Record<string, unknown> = {
    document_id: docId,
    platform,
    native_id: id,
    url: `https://${platform}.com/${id}`,
    timestamps: {
      published_at: '2026-11-11T08:48:24+00:00',
      collected_at: '2026-11-11T08:48:24+00:00',
      processed_at: '2026-11-11T08:48:24+00:00',
    },
    author: {
      username: `user_${id}`,
      channel_id: platform === 'youtube' ? `channel_${id}` : null,
      channel_title: platform === 'youtube' ? `Channel ${id}` : null,
      graph_status: 'ranked_influencer',
      influencer_score: 0.85,
      pagerank: 0.012,
      community_id: 1,
      in_degree: 5,
      out_degree: 2,
    },
    content: {
      title_raw: `Title ${id}`,
      body_raw: `Body content for ${id}`,
      title_cleaned: `Title ${id}`,
      body_cleaned: `Body content for ${id}`,
      detected_language: 'en',
      language_code: 'en',
      preprocessing_status: 'completed',
    },
    engagement: {
      score: 10,
      views_count: null,
      likes_count: null,
      comments_count: 2,
    },
    metadata: {
      reddit: platform === 'reddit' ? { subreddit: 'news', score: 10 } : null,
      youtube: platform === 'youtube' ? { channel_id: `channel_${id}`, channel_title: `Channel ${id}` } : null,
    },
    consolidation: {
      status: 'complete',
      is_complete: true,
      sources_joined: [platform],
    },
    sentiment: {
      status: 'success',
      label: 'positive',
      score: 0.75,
    },
    topics: {
      status: 'assigned',
      topic_id: 1,
      probability: 0.85,
      keywords: [{ word: 'Technology', weight: 0.85 }],
    },
    misinformation: {
      status: 'success',
      veracity_label: 'real',
      credibility_score: 0.82,
      risk_tier: 'low',
      signals_detected: ['consistent_timing'],
      stylometric_signals: {
        lexical_richness: 0.65,
        sentiment_entropy: 0.42,
      },
    },
    ...overrides,
  };
  return base as unknown as ModernCanonicalRecord;
}

function createSampleModernConsolidated(
  records: ModernCanonicalRecord[]
): ValidatedModernConsolidated {
  const redditCount = records.filter((r) => r.platform === 'reddit').length;
  const youtubeCount = records.filter((r) => r.platform === 'youtube').length;
  const completeCount = records.filter((r) => r.consolidation?.is_complete).length;

  return {
    schema_version: '2.0',
    pipeline_version: '2.0.0',
    status: 'success',
    consolidation_status: 'complete',
    generated_at: '2026-11-11T08:48:24+00:00',
    provenance: {
      reddit_input_file: 'reddit_data.json',
      reddit_status: 'loaded',
      reddit_records_count: redditCount,
      youtube_input_file: 'youtube_data.json',
      youtube_status: youtubeCount > 0 ? 'loaded' : 'missing',
      youtube_records_count: youtubeCount,
    },
    metrics: {
      total_records_consolidated: records.length,
      platform_counts: {
        reddit: redditCount,
        youtube: youtubeCount,
      },
      complete_records: completeCount,
      degraded_records: 0,
      partial_records: 0,
      unprocessed_records: records.length - completeCount,
      duplicates_deduplicated: 0,
      rejected_records_count: 0,
    },
    audit_warnings: [],
    rejected_records: [],
    records,
  };
}

// Suite 1: Filter Recalculation

test('filterCanonicalRecords: returns all records when filter is "all"', () => {
  const rec1 = createSampleRecord('1', 'reddit');
  const rec2 = createSampleRecord('2', 'youtube');
  const records = [rec1, rec2];

  const result = filterCanonicalRecords(records, 'all');
  assert.equal(result.length, 2);
  assert.notEqual(result, records, 'Should return a shallow copy, not mutate original');
});

test('filterCanonicalRecords: filters strictly by reddit or youtube', () => {
  const rec1 = createSampleRecord('1', 'reddit');
  const rec2 = createSampleRecord('2', 'youtube');
  const rec3 = createSampleRecord('3', 'reddit');
  const records = [rec1, rec2, rec3];

  const redditOnly = filterCanonicalRecords(records, 'reddit');
  assert.equal(redditOnly.length, 2);
  assert.ok(redditOnly.every((r) => r.platform === 'reddit'));

  const youtubeOnly = filterCanonicalRecords(records, 'youtube');
  assert.equal(youtubeOnly.length, 1);
  assert.equal(youtubeOnly[0].document_id, 'youtube_2');
});

test('filterCanonicalRecords: gracefully handles empty or invalid inputs', () => {
  assert.deepEqual(filterCanonicalRecords([], 'all'), []);
  assert.deepEqual(filterCanonicalRecords(null as unknown as ModernCanonicalRecord[], 'reddit'), []);
  assert.deepEqual(filterCanonicalRecords(undefined as unknown as ModernCanonicalRecord[], 'youtube'), []);
});

// Suite 2: Executive Overview Analytics

test('calculateOverviewMetrics: separates envelope metrics from record metrics under filtering', () => {
  const rec1 = createSampleRecord('1', 'reddit', {
    consolidation: { status: 'complete', is_complete: true, sources_joined: ['reddit'] },
  });
  const rec2 = createSampleRecord('2', 'reddit', {
    consolidation: { status: 'unprocessed', is_complete: false, sources_joined: ['reddit'] },
  });
  const rec3 = createSampleRecord('3', 'youtube', {
    consolidation: { status: 'complete', is_complete: true, sources_joined: ['youtube'] },
  });

  const allRecords = [rec1, rec2, rec3];
  const payload = createSampleModernConsolidated(allRecords);

  // When filtered to reddit
  const redditFiltered = filterCanonicalRecords(allRecords, 'reddit');
  const metrics = calculateOverviewMetrics(payload, redditFiltered, null, null);

  // Envelope metrics remain whole-run fixed scope
  assert.equal(metrics.totalRecordsConsolidated, 3);
  assert.equal(metrics.redditCount, 2);
  assert.equal(metrics.youtubeCount, 1);
  assert.equal(metrics.completeRecordsRun, 2);
  assert.equal(metrics.unprocessedRecordsRun, 1);

  // Record metrics recalculate for active filter
  assert.equal(metrics.filteredRecordsCount, 2);
  assert.equal(metrics.filteredCompleteCount, 1);
  assert.equal(metrics.filteredCompletePct, 50.0);
  assert.equal(metrics.filteredSourcesJoined.reddit, 2);
  assert.equal(metrics.filteredSourcesJoined.youtube, undefined);
});

test('calculateOverviewMetrics: extracts YouTube ingestion status when count is 0', () => {
  const records = [createSampleRecord('1', 'reddit')];
  const payload = createSampleModernConsolidated(records);

  const manifest = {
    stages: {
      youtube_ingestion: {
        status: 'skipped',
        reason: 'YouTube API key not configured',
      },
    },
  };

  const metrics = calculateOverviewMetrics(payload, records, null, manifest);
  assert.equal(metrics.youtubeCount, 0);
  assert.equal(metrics.youtubeStatusDetail, '[skipped] YouTube API key not configured');
});

test('calculateOverviewMetrics: returns null complete percentage when filtered records count is 0', () => {
  const records = [createSampleRecord('1', 'reddit')];
  const payload = createSampleModernConsolidated(records);

  const emptyFilter: ModernCanonicalRecord[] = [];
  const metrics = calculateOverviewMetrics(payload, emptyFilter, null, null);

  assert.equal(metrics.filteredRecordsCount, 0);
  assert.equal(metrics.filteredCompleteCount, 0);
  assert.equal(metrics.filteredCompletePct, null);
});

// Suite 3: Sentiment Analysis Analytics

test('calculateSentimentMetrics: only counts success positive/neutral/negative as assessed', () => {
  const r1 = createSampleRecord('1', 'reddit', { sentiment: { status: 'success', label: 'positive', score: 0.8 } });
  const r2 = createSampleRecord('2', 'reddit', { sentiment: { status: 'success', label: 'neutral', score: 0.0 } });
  const r3 = createSampleRecord('3', 'reddit', { sentiment: { status: 'success', label: 'negative', score: -0.5 } });
  const r4 = createSampleRecord('4', 'reddit', { sentiment: { status: 'success', label: 'undetermined', score: 0 } });
  const r5 = createSampleRecord('5', 'reddit', { sentiment: { status: 'failed', label: 'positive', score: 0.9 } });
  const r6 = createSampleRecord('6', 'reddit', { sentiment: null });
  const r7 = createSampleRecord('7', 'reddit', { sentiment: { status: 'success', label: 'invalid_label', score: 0.1 } });

  const result = calculateSentimentMetrics([r1, r2, r3, r4, r5, r6, r7]);

  assert.equal(result.totalRecordsEvaluated, 7);
  assert.equal(result.assessedCount, 3);
  assert.equal(result.undeterminedCount, 1);
  assert.equal(result.unassessedCount, 3); // r5 (status failed) + r6 (null) + r7 (unexpected label)

  assert.equal(result.distribution.positive, 1);
  assert.equal(result.distribution.neutral, 1);
  assert.equal(result.distribution.negative, 1);

  // Percentages calculated over assessed count (3)
  assert.equal(result.distribution.positivePct, 33.3);
  assert.equal(result.distribution.neutralPct, 33.3);
  assert.equal(result.distribution.negativePct, 33.3);

  // Net sentiment: (1 - 1) / 3 = 0.0
  assert.equal(result.netSentiment, 0.0);
});

test('calculateSentimentMetrics: returns netSentiment null when assessedCount is 0', () => {
  const r1 = createSampleRecord('1', 'reddit', { sentiment: { status: 'success', label: 'undetermined', score: 0 } });
  const r2 = createSampleRecord('2', 'reddit', { sentiment: { status: 'failed', label: 'neutral', score: 0 } });
  const r3 = createSampleRecord('3', 'reddit', { sentiment: null });

  const result = calculateSentimentMetrics([r1, r2, r3]);

  assert.equal(result.assessedCount, 0);
  assert.equal(result.undeterminedCount, 1);
  assert.equal(result.unassessedCount, 2);
  assert.equal(result.netSentiment, null, 'Net sentiment MUST be null when 0 assessed records exist, never 0.0');
  assert.equal(result.distribution.positivePct, null);
  assert.equal(result.distribution.neutralPct, null);
  assert.equal(result.distribution.negativePct, null);
});

test('calculateSentimentMetrics: populates score histogram bins across [-1.0, 1.0]', () => {
  const records = [
    createSampleRecord('1', 'reddit', { sentiment: { status: 'success', label: 'negative', score: -0.8 } }), // bin 0: [-1.0, -0.6)
    createSampleRecord('2', 'reddit', { sentiment: { status: 'success', label: 'negative', score: -0.4 } }), // bin 1: [-0.6, -0.2)
    createSampleRecord('3', 'reddit', { sentiment: { status: 'success', label: 'neutral', score: 0.0 } }),   // bin 2: [-0.2, 0.2]
    createSampleRecord('4', 'reddit', { sentiment: { status: 'success', label: 'positive', score: 0.5 } }),  // bin 3: (0.2, 0.6]
    createSampleRecord('5', 'reddit', { sentiment: { status: 'success', label: 'positive', score: 0.9 } }),  // bin 4: (0.6, 1.0]
  ];

  const result = calculateSentimentMetrics(records);
  assert.equal(result.scoreHistogram.counts[0], 1);
  assert.equal(result.scoreHistogram.counts[1], 1);
  assert.equal(result.scoreHistogram.counts[2], 1);
  assert.equal(result.scoreHistogram.counts[3], 1);
  assert.equal(result.scoreHistogram.counts[4], 1);
});

test('calculateSentimentMetrics: calculates platform breakdown metrics separately', () => {
  const records = [
    createSampleRecord('1', 'reddit', { sentiment: { status: 'success', label: 'positive', score: 0.8 } }),
    createSampleRecord('2', 'reddit', { sentiment: { status: 'success', label: 'positive', score: 0.7 } }),
    createSampleRecord('3', 'youtube', { sentiment: { status: 'success', label: 'negative', score: -0.6 } }),
  ];

  const result = calculateSentimentMetrics(records);
  assert.equal(result.platformBreakdown.reddit.assessed, 2);
  assert.equal(result.platformBreakdown.reddit.netSentiment, 1.0);
  assert.equal(result.platformBreakdown.reddit.positivePct, 100.0);

  assert.equal(result.platformBreakdown.youtube.assessed, 1);
  assert.equal(result.platformBreakdown.youtube.netSentiment, -1.0);
  assert.equal(result.platformBreakdown.youtube.positivePct, 0.0);
});

// Suite 4: Topic Intelligence Analytics

test('getTopicUnavailableMessage: extracts verbatim reason or error_category or generic fallback', () => {
  // Case 1: Non-empty reason returned verbatim without prefix
  const manifestWithReason = {
    stages: {
      topics: {
        status: 'failed',
        reason: 'Topic execution encountered an unexpected error',
      },
    },
  };
  const msg1 = getTopicUnavailableMessage(null, null, manifestWithReason);
  assert.equal(msg1, 'Topic execution encountered an unexpected error');

  // Case 2: Non-empty error_category returned verbatim when reason is absent
  const manifestWithErrorCat = {
    stages: {
      topics: {
        status: 'failed',
        error_category: 'execution_error',
      },
    },
  };
  const msg2 = getTopicUnavailableMessage(null, null, manifestWithErrorCat);
  assert.equal(msg2, 'execution_error');

  // Case 3: Status-only manifest uses generic fallback
  const manifestWithStatusOnly = {
    stages: {
      topics: {
        status: 'skipped',
      },
    },
  };
  const msg3 = getTopicUnavailableMessage(null, null, manifestWithStatusOnly);
  assert.equal(msg3, 'Topic artifact unavailable for this dataset run');

  // Case 4: String-valued stage uses generic fallback
  const stagesWithString = { topics: 'failed' };
  const msg4 = getTopicUnavailableMessage(null, stagesWithString, null);
  assert.equal(msg4, 'Topic artifact unavailable for this dataset run');

  // Case 5: Null / undefined manifest uses generic fallback
  const msg5 = getTopicUnavailableMessage(null, null, null);
  assert.equal(msg5, 'Topic artifact unavailable for this dataset run');

  // Case 6: Preserves leading and trailing whitespace verbatim in reason
  const manifestWithSpacedReason = {
    stages: {
      topics: {
        status: 'failed',
        reason: '  Topic error with leading and trailing whitespace  ',
      },
    },
  };
  const msg6 = getTopicUnavailableMessage(null, null, manifestWithSpacedReason);
  assert.equal(msg6, '  Topic error with leading and trailing whitespace  ', 'Must preserve original whitespace without trimming');

  // Case 7: Preserves leading and trailing whitespace verbatim in error_category
  const manifestWithSpacedCategory = {
    stages: {
      topics: {
        status: 'failed',
        error_category: '  execution_timeout  ',
      },
    },
  };
  const msg7 = getTopicUnavailableMessage(null, null, manifestWithSpacedCategory);
  assert.equal(msg7, '  execution_timeout  ', 'Must preserve original category whitespace without trimming');

  // Case 8: Whitespace-only reason falls back to error_category if present
  const manifestWithWhitespaceReason = {
    stages: {
      topics: {
        status: 'failed',
        reason: '   \t\n  ',
        error_category: 'fallback_category',
      },
    },
  };
  const msg8 = getTopicUnavailableMessage(null, null, manifestWithWhitespaceReason);
  assert.equal(msg8, 'fallback_category', 'Whitespace-only reason must allow fallback to valid error_category');

  // Case 9: Empty and whitespace-only reason and error_category use generic fallback
  const manifestWithWhitespaceBoth = {
    stages: {
      topics: {
        status: 'failed',
        reason: '   ',
        error_category: '  \n\t  ',
      },
    },
  };
  const msg9 = getTopicUnavailableMessage(null, null, manifestWithWhitespaceBoth);
  assert.equal(msg9, 'Topic artifact unavailable for this dataset run', 'Whitespace-only diagnostics must trigger generic fallback');
});

test('extractTopicMetrics: handles unavailable artifact defensively with verbatim message', () => {
  const manifest = {
    stages: {
      topics: {
        status: 'failed',
        reason: 'Topic execution encountered an unexpected error',
      },
    },
  };

  const result = extractTopicMetrics(null, [], null, manifest);
  assert.equal(result.isAvailable, false);
  assert.equal(result.unavailableReason, 'Topic execution encountered an unexpected error');
  assert.equal(result.totalDocumentsModeled, null);
  assert.equal(result.outlierCount, 0);
  assert.deepEqual(result.topics, []);
});

test('extractTopicMetrics: parses valid topic modeling artifact and excludes outlier from topic list', () => {
  const artifact: TopicModelingArtifact = {
    status: 'success',
    generated_at: '2026-11-11T08:48:24+00:00',
    embedding_model: 'all-MiniLM-L6-v2',
    total_documents_modeled: 45,
    outlier_count: 5,
    topic_info: [
      {
        topic_id: -1,
        name: '-1_outlier_noise',
        count: 5,
        keywords: ['noise', 'spam'],
      },
      {
        topic_id: 0,
        name: '0_ai_machine_learning',
        count: 25,
        keywords: [
          { word: 'ai', weight: 0.8 },
          { word: 'learning', weight: 0.6 },
        ],
      },
      {
        topic_id: 1,
        name: '1_space_astronomy',
        count: 15,
        keywords: ['telescope', 'galaxy'],
      },
    ],
  };

  const result = extractTopicMetrics(artifact, []);
  assert.equal(result.isAvailable, true);
  assert.equal(result.unavailableReason, null);
  assert.equal(result.totalDocumentsModeled, 45);
  assert.equal(result.outlierCount, 5);

  // Outlier topic -1 must NOT be in clusters list
  assert.equal(result.topics.length, 2);
  assert.equal(result.topics[0].topicId, 0);
  assert.equal(result.topics[0].keywords[0].word, 'ai');
  assert.equal(result.topics[0].keywords[0].weight, 0.8);

  assert.equal(result.topics[1].topicId, 1);
  assert.equal(result.topics[1].keywords[0].word, 'telescope');
  assert.equal(result.topics[1].keywords[0].weight, null);
});

// Suite 5: Misinformation Risk Analytics

test('calculateMisinformationMetrics: strictly excludes veracity_label from all outputs', () => {
  const rec = createSampleRecord('1', 'reddit', {
    misinformation: {
      status: 'success',
      credibility_score: 0.85,
      risk_tier: 'low',
      veracity_label: 'real', // Present in raw record schema
      signals_detected: ['timing_regular'],
      stylometric_signals: { lexical_richness: 0.7 },
    },
  });

  const result = calculateMisinformationMetrics([rec]);

  // Assert veracity_label is NOT exposed in the analytics output structure
  assert.equal('veracity_label' in (result as unknown as Record<string, unknown>), false);
  assert.equal('veracity' in (result as unknown as Record<string, unknown>), false);
  assert.equal('veracity_breakdown' in (result as unknown as Record<string, unknown>), false);

  const serialized = JSON.stringify(result);
  assert.equal(serialized.includes('veracity_label'), false);
  assert.equal(serialized.includes('"real"'), false);
});

test('calculateMisinformationMetrics: maps risk tiers according to canonical thresholds', () => {
  const rHigh = createSampleRecord('1', 'reddit', {
    misinformation: { status: 'success', credibility_score: 0.35, risk_tier: 'high_risk' },
  });
  const rMod = createSampleRecord('2', 'reddit', {
    misinformation: { status: 'success', credibility_score: 0.55, risk_tier: 'moderate' },
  });
  const rLow = createSampleRecord('3', 'reddit', {
    misinformation: { status: 'success', credibility_score: 0.85, risk_tier: 'low' },
  });
  const rFailed = createSampleRecord('4', 'reddit', {
    misinformation: { status: 'failed', credibility_score: 0.1, risk_tier: 'high_risk' },
  });

  const result = calculateMisinformationMetrics([rHigh, rMod, rLow, rFailed]);

  assert.equal(result.assessedCount, 3);
  assert.equal(result.unassessedCount, 1);
  assert.equal(result.riskTiers.highRisk, 1);
  assert.equal(result.riskTiers.moderate, 1);
  assert.equal(result.riskTiers.low, 1);

  // Average credibility score: (0.35 + 0.55 + 0.85) / 3 = 1.75 / 3 = 0.583
  assert.equal(result.averageCredibilityScore, 0.583);
});

test('calculateMisinformationMetrics: strictly excludes records with missing or invalid credibility scores from risk distribution', () => {
  const rValidHigh = createSampleRecord('1', 'reddit', {
    misinformation: { status: 'success', credibility_score: 0.35, risk_tier: 'high_risk' },
  });
  // Record with null credibility score but stored risk_tier string
  const rNullScoreWithTier = createSampleRecord('2', 'reddit', {
    misinformation: { status: 'success', credibility_score: null, risk_tier: 'moderate' },
  });
  // Record with invalid out-of-range credibility score (-1.5) but stored risk_tier string
  const rNegativeScore = createSampleRecord('3', 'reddit', {
    misinformation: { status: 'success', credibility_score: -1.5, risk_tier: 'high_risk' },
  });
  // Record with invalid out-of-range credibility score (2.0) but stored risk_tier string
  const rExceededScore = createSampleRecord('4', 'reddit', {
    misinformation: { status: 'success', credibility_score: 2.0, risk_tier: 'low' },
  });
  // Record with non-numeric credibility score
  const rNaNScore = createSampleRecord('5', 'reddit', {
    misinformation: { status: 'success', credibility_score: 'not-a-number' as any, risk_tier: 'low' },
  });

  const result = calculateMisinformationMetrics([
    rValidHigh,
    rNullScoreWithTier,
    rNegativeScore,
    rExceededScore,
    rNaNScore,
  ]);

  // Total assessed count increments for all status: success
  assert.equal(result.assessedCount, 5);

  // But risk tiers MUST ONLY include records with valid finite scores in [0.0, 1.0]!
  // Stored risk_tier must NOT silently compensate for missing or invalid scores.
  assert.equal(result.riskTiers.highRisk, 1, 'Only rValidHigh should be counted in highRisk');
  assert.equal(result.riskTiers.moderate, 0, 'rNullScoreWithTier must be excluded from moderate tier');
  assert.equal(result.riskTiers.low, 0, 'Invalid score records must be excluded from low tier');

  // Average credibility score only considers the single valid score (0.35)
  assert.equal(result.averageCredibilityScore, 0.35);

  // Histogram counts must also strictly reflect only the single valid score
  assert.equal(result.scoreHistogram.counts[0], 1);
  assert.equal(result.scoreHistogram.counts[1], 0);
  assert.equal(result.scoreHistogram.counts[2], 0);
  assert.equal(result.scoreHistogram.counts[3], 0);
});

test('calculateMisinformationMetrics: aggregates genuine discourse flags and stylometric signals', () => {
  const r1 = createSampleRecord('1', 'reddit', {
    misinformation: {
      status: 'success',
      credibility_score: 0.6,
      risk_tier: 'moderate',
      signals_detected: ['emotional_charge', 'hyperbolic_language'],
      stylometric_signals: { lexical_richness: 0.5, readability_index: 60.0 },
    },
  });
  const r2 = createSampleRecord('2', 'reddit', {
    misinformation: {
      status: 'success',
      credibility_score: 0.7,
      risk_tier: 'low',
      signals_detected: ['emotional_charge', ''], // Empty string should be ignored
      stylometric_signals: { lexical_richness: 0.7 },
    },
  });
  const r3 = createSampleRecord('3', 'reddit', {
    misinformation: {
      status: 'failed',
      signals_detected: ['fake_signal'], // Should not be aggregated because status is failed
    },
  });

  const result = calculateMisinformationMetrics([r1, r2, r3]);

  // Verify aggregated signals
  const flagMap = Object.fromEntries(result.detectedSignals.map((s) => [s.flag, s.count]));
  assert.equal(flagMap.emotional_charge, 2);
  assert.equal(flagMap.hyperbolic_language, 1);
  assert.equal(flagMap[''], undefined);
  assert.equal(flagMap.fake_signal, undefined);

  // Verify stylometric averages with independent denominators
  assert.equal(result.stylometricAverages.lexical_richness, 0.6); // (0.5 + 0.7) / 2
  assert.equal(result.stylometricAverages.readability_index, 60.0); // 60 / 1

  // Verify epistemic disclosure text matches standard requirement
  assert.equal(result.epistemicDisclosure, METHODOLOGICAL_DISCLOSURE_TEXT);
});

// Suite 6: Network & Influencer Analytics

test('extractNetworkMetrics: accurately reflects 10 disconnected nodes with 0 edges from schema metadata', () => {
  const artifact: GraphAnalysisArtifact = {
    status: 'success',
    generated_at: '2026-11-11T08:48:24+00:00',
    nodes: [
      { node_id: 'user_1', platform: 'reddit', identity_type: 'account', username: 'user_1', activity_count: 5 },
      { node_id: 'user_2', platform: 'reddit', identity_type: 'account', username: 'user_2', activity_count: 3 },
      { node_id: 'user_3', platform: 'reddit', identity_type: 'account', username: 'user_3', activity_count: 2 },
      { node_id: 'user_4', platform: 'reddit', identity_type: 'account', username: 'user_4', activity_count: 2 },
      { node_id: 'user_5', platform: 'reddit', identity_type: 'account', username: 'user_5', activity_count: 1 },
      { node_id: 'user_6', platform: 'reddit', identity_type: 'account', username: 'user_6', activity_count: 1 },
      { node_id: 'user_7', platform: 'reddit', identity_type: 'account', username: 'user_7', activity_count: 1 },
      { node_id: 'user_8', platform: 'reddit', identity_type: 'account', username: 'user_8', activity_count: 1 },
      { node_id: 'user_9', platform: 'reddit', identity_type: 'account', username: 'user_9', activity_count: 1 },
      { node_id: 'user_10', platform: 'reddit', identity_type: 'account', username: 'user_10', activity_count: 1 },
    ],
    edges: [],
    graph_metrics: {
      total_users: 10,
      total_interactions: 0,
      density: 0.0,
      communities_count: 10,
      is_directed: true,
    },
    top_influencers: [
      {
        node_id: 'user_1',
        username: 'user_1',
        platform: 'reddit',
        influencer_score: 0.5,
        pagerank: 0.1,
        in_degree: 0,
        out_degree: 0,
        in_degree_raw: 0,
        out_degree_raw: 0,
      },
    ],
  };

  const result = extractNetworkMetrics(artifact);

  assert.equal(result.isAvailable, true);
  assert.equal(result.totalUsers, 10);
  assert.equal(result.totalInteractions, 0);
  assert.equal(result.density, 0.0);
  assert.equal(result.communitiesCount, 10);
  assert.equal(result.nodes.length, 10);

  // Assert node schema mapping
  assert.equal(result.nodes[0].nodeId, 'user_1');
  assert.equal(result.nodes[0].identityType, 'account');
  assert.equal(result.nodes[0].platform, 'reddit');
  assert.equal(result.nodes[0].activityCount, 5);

  // Assert influencer mapping
  assert.equal(result.influencers.length, 1);
  assert.equal(result.influencers[0].username, 'user_1');
  assert.equal(result.influencers[0].inDegree, 0);
  assert.equal(result.influencers[0].outDegree, 0);
});

test('extractNetworkMetrics: handles null artifact gracefully', () => {
  const result = extractNetworkMetrics(null);

  assert.equal(result.isAvailable, false);
  assert.equal(result.totalUsers, 0);
  assert.equal(result.totalInteractions, 0);
  assert.equal(result.density, 0);
  assert.deepEqual(result.nodes, []);
  assert.deepEqual(result.influencers, []);
});

test('derivePlatformTopicMetrics: mathematically derives clusters from individual records', () => {
  const redditRec1 = createSampleRecord('1', 'reddit', {
    topics: {
      status: 'assigned',
      topic_id: 1,
      probability: 0.85,
      keywords: [{ word: 'AI', weight: 0.9 }, { word: 'Model', weight: 0.8 }],
    },
  });
  const redditRec2 = createSampleRecord('2', 'reddit', {
    topics: {
      status: 'assigned',
      topic_id: 1,
      probability: 0.85,
      keywords: [{ word: 'AI', weight: 0.7 }, { word: 'Ethics', weight: 0.6 }],
    },
  });
  const redditRec3 = createSampleRecord('3', 'reddit', {
    topics: {
      status: 'assigned',
      topic_id: 2,
      probability: 0.85,
      keywords: [{ word: 'Cloud', weight: 0.95 }],
    },
  });

  const result = derivePlatformTopicMetrics([redditRec1, redditRec2, redditRec3]);

  assert.equal(result.isAvailable, true);
  assert.equal(result.totalDocumentsModeled, 3);
  assert.equal(result.topics.length, 2);
  assert.equal(result.topics[0].topicId, 1);
  assert.equal(result.topics[0].count, 2);
  // Verify keyword aggregation order (AI has mean (0.9+0.7)/2 = 0.8)
  assert.equal(result.topics[0].keywords[0].word, 'AI');
  assert.equal(result.topics[0].keywords[0].weight, 0.8);

  assert.equal(result.topics[1].topicId, 2);
  assert.equal(result.topics[1].count, 1);
});

test('derivePlatformTopicMetrics: returns empty state when platform has 0 records', () => {
  const result = derivePlatformTopicMetrics([], null, 'youtube');
  assert.equal(result.isAvailable, false);
  assert.equal(result.emptyStateReason, 'empty_platform');
  assert.equal(result.topics.length, 0);
  assert.equal(result.totalDocumentsModeled, 0);
});

test('derivePlatformGraphMetrics: filters nodes and enforces edge endpoint consistency', () => {
  const artifact: GraphAnalysisArtifact = {
    schema_version: '2.0',
    status: 'success',
    generated_at: '2026-11-14T10:15:00Z',
    graph_metrics: {
      total_users: 3,
      total_interactions: 2,
      density: 0.33,
      communities_count: 1,
      is_directed: true,
    },
    nodes: [
      { node_id: 'reddit:user:alice', platform: 'reddit', username: 'alice', activity_count: 2 },
      { node_id: 'reddit:user:bob', platform: 'reddit', username: 'bob', activity_count: 2 },
      { node_id: 'youtube:channel:charlie', platform: 'youtube', username: 'charlie', activity_count: 1 },
    ],
    edges: [
      { source: 'reddit:user:alice', target: 'reddit:user:bob', weight: 1 },
      { source: 'reddit:user:alice', target: 'youtube:channel:charlie', weight: 1 },
    ],
    top_influencers: [
      { node_id: 'reddit:user:bob', platform: 'reddit', in_degree_raw: 1, out_degree_raw: 0, influencer_score: 0.8 },
      { node_id: 'reddit:user:isolated', platform: 'reddit', in_degree_raw: 0, out_degree_raw: 0, influencer_score: 0.5 },
    ],
  };

  // 1. All platforms
  const allResult = derivePlatformGraphMetrics(artifact, 'all');
  assert.equal(allResult.totalUsers, 3);
  assert.equal(allResult.totalInteractions, 2);

  // 2. Reddit only: Charlie is dropped, and edge alice -> charlie must be dropped!
  const redditResult = derivePlatformGraphMetrics(artifact, 'reddit');
  assert.equal(redditResult.totalUsers, 2);
  assert.equal(redditResult.totalInteractions, 1); // Only alice -> bob retained
  assert.equal(redditResult.nodes.every((n) => n.platform === 'reddit'), true);

  // 3. YouTube only: 1 node (charlie), 0 edges (incoming edge from alice dropped)
  const ytResult = derivePlatformGraphMetrics(artifact, 'youtube');
  assert.equal(ytResult.totalUsers, 1);
  assert.equal(ytResult.totalInteractions, 0);
  assert.equal(ytResult.influencers.length, 0); // Suppressed isolated account
});

test('getPlatformEmptyState: generates honest manifest-driven messages', () => {
  const ytState = getPlatformEmptyState('youtube', 0, {
    youtube_ingestion: { status: 'skipped', reason: 'youtube_api_key_not_configured' },
  });
  assert.equal(ytState.isEmpty, true);
  assert.equal(ytState.manifestReason, 'youtube_api_key_not_configured');
  assert.match(ytState.reason, /YouTube Data API key was not configured/);

  const populatedState = getPlatformEmptyState('reddit', 10);
  assert.equal(populatedState.isEmpty, false);
});

test('filterAndSortFeedRecords: case-insensitive search and multidimensional filters', () => {
  const records = [
    {
      document_id: 'reddit_post_001',
      platform: 'reddit',
      author: { username: 'alice_researcher' },
      content: { title_cleaned: 'Deep dive into RoBERTa fine-tuning', body_cleaned: 'Transformers are effective for NLP.' },
      sentiment: { status: 'success', label: 'positive', score: 0.8 },
      topics: { topic_id: 0 },
      timestamps: { published_at: '2026-11-10T12:00:00Z' },
      engagement: { score: 100 },
      misinformation: { credibility_score: 0.9, risk_tier: 'low' },
    },
    {
      document_id: 'reddit_post_002',
      platform: 'reddit',
      author: { username: 'bob_dev' },
      content: { title_cleaned: 'Benchmark evaluation dataset analysis', body_cleaned: 'Reviewing cross-lingual performance.' },
      sentiment: { status: 'fallback', label: null, score: null },
      topics: { topic_id: 1 },
      timestamps: { published_at: '2026-11-12T15:30:00Z' },
      engagement: { score: 45 },
      misinformation: { credibility_score: 0.65, risk_tier: 'moderate' },
    },
    {
      document_id: 'youtube_video_001',
      platform: 'youtube',
      author: { username: 'tech_channel', channel_title: 'Tech Explained' },
      content: { title_cleaned: 'Is this AI news misleading?', body_cleaned: 'Examining clickbait titles on social media.' },
      sentiment: { status: 'success', label: 'negative', score: -0.6 },
      topics: { topic_id: -1 },
      timestamps: { published_at: '2026-11-14T09:00:00Z' },
      engagement: { score: 250 },
      misinformation: { credibility_score: 0.4, risk_tier: 'high_risk' },
    },
  ] as unknown as ModernCanonicalRecord[];

  // 1. Text search for "RoBERTa"
  const search1 = filterAndSortFeedRecords(records, { query: 'roberta' });
  assert.equal(search1.length, 1);
  assert.equal(search1[0].document_id, 'reddit_post_001');

  // 2. Search for author channel "Tech Explained"
  const search2 = filterAndSortFeedRecords(records, { query: 'Tech Explained' });
  assert.equal(search2.length, 1);
  assert.equal(search2[0].document_id, 'youtube_video_001');

  // 3. Sentiment filter for undetermined
  const undetermined = filterAndSortFeedRecords(records, { sentimentFilter: 'undetermined' });
  assert.equal(undetermined.length, 1);
  assert.equal(undetermined[0].document_id, 'reddit_post_002');

  // 4. Topic filter for outlier (-1)
  const outliers = filterAndSortFeedRecords(records, { topicFilter: 'outlier' });
  assert.equal(outliers.length, 1);
  assert.equal(outliers[0].document_id, 'youtube_video_001');

  // 5. Sort by newest
  const sortedNewest = filterAndSortFeedRecords(records, { sortBy: 'newest' });
  assert.equal(sortedNewest[0].document_id, 'youtube_video_001'); // Nov 14
  assert.equal(sortedNewest[2].document_id, 'reddit_post_001'); // Nov 10

  // 6. Sort by engagement score
  const sortedScore = filterAndSortFeedRecords(records, { sortBy: 'score' });
  assert.equal(sortedScore[0].document_id, 'youtube_video_001'); // 250
  assert.equal(sortedScore[1].document_id, 'reddit_post_001'); // 100
  assert.equal(sortedScore[2].document_id, 'reddit_post_002'); // 45
});
