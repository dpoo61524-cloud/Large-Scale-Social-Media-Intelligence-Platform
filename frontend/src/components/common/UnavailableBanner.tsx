import React from 'react';

interface UnavailableBannerProps {
  title?: string;
  reason?: string | null;
  defaultReason?: string;
}

export const UnavailableBanner: React.FC<UnavailableBannerProps> = ({
  title = 'Artifact Unavailable',
  reason,
  defaultReason = 'This analytical artifact was not produced in the selected dataset run.',
}) => {
  const displayText = (typeof reason === 'string' && reason.trim().length > 0) ? reason : defaultReason;

  return (
    <div className="unavailable-banner">
      <div className="unavailable-title">{title}</div>
      <div className="unavailable-reason">{displayText}</div>
    </div>
  );
};

