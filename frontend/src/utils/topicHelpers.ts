// Topic Classification & Theme Formatting Utilities
// Light Yellow & Dark Forest Green Theme Harmonization

export interface CleanTopicMeta {
  title: string;
  shortLabel: string;
  themeColor: string;
  badgeBg: string;
}

/**
 * Returns descriptive, human-understandable titles and theme colors for topic clusters.
 * Based on concrete post contents from the ingested Reddit and YouTube datasets.
 */
export function getTopicMeta(rawName: string, topicId?: number): CleanTopicMeta {
  const tid = topicId !== undefined ? topicId : -1;
  const lower = (rawName || '').toLowerCase();

  if (tid === 0 || lower.includes('streaming')) {
    return {
      title: 'Real-Time Streaming Engine Benchmarks',
      shortLabel: 'Streaming Benchmarks',
      themeColor: '#064E3B', // Primary Dark Forest Green
      badgeBg: '#FEF9C3',
    };
  }

  if (tid === 1 || lower.includes('regarding production') || lower.includes('production')) {
    return {
      title: 'Production Architecture & Deployment',
      shortLabel: 'Production Systems',
      themeColor: '#047857', // Deep Emerald Green
      badgeBg: '#D1FAE5',
    };
  }

  if (tid === 2 || lower.includes('intelligence')) {
    return {
      title: 'AI & Machine Learning Educational Lectures',
      shortLabel: 'AI Education & ML',
      themeColor: '#0D9488', // Teal Pine
      badgeBg: '#CCFBF1',
    };
  }

  if (tid === 3 || lower.includes('continuing')) {
    return {
      title: 'Data Preprocessing & Text Cleaning Discussions',
      shortLabel: 'Text Preprocessing',
      themeColor: '#2D6A4F', // Forest Sage
      badgeBg: '#E8F5E9',
    };
  }

  if (tid === 4 || lower.includes('ai')) {
    return {
      title: 'Social Media Analytics & Graph Intelligence',
      shortLabel: 'Social Graph AI',
      themeColor: '#B45309', // Warm Amber Gold
      badgeBg: '#FEF3C7',
    };
  }

  // Fallback
  const stripped = (rawName || '')
    .replace(/^\d+[_]/, '')
    .replace(/[_]+/g, ' ')
    .trim();

  const formatted = stripped
    ? stripped
        .split(' ')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ')
    : `Topic #${topicId ?? ''}`;

  return {
    title: formatted,
    shortLabel: formatted.length > 22 ? formatted.slice(0, 20) + '...' : formatted,
    themeColor: '#064E3B',
    badgeBg: '#FEF9C3',
  };
}

/**
 * Cleans keyword strings from n-gram artifacts (e.g. 'continuing _' -> 'continuing', 'threat_hunter_zoe' -> 'threat hunter zoe')
 */
export function cleanKeyword(word: string): string {
  if (!word) return '';
  return word
    .replace(/[_]+/g, ' ')
    .trim();
}
