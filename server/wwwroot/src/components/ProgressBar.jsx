import React from 'react';

/**
 * Horizontal progress bar.
 * Props:
 *   value  – completion percentage 0–100 (default 0)
 *   label  – optional text shown above the bar
 */
const ProgressBar = ({ value = 0, label }) => {
    const pct = Math.min(100, Math.max(0, value));
    return (
        <div style={{ width: '100%' }}>
            {label && (
                <div style={{ marginBottom: '0.3rem', fontSize: '0.85rem', color: '#374151' }}>
                    {label}
                </div>
            )}
            <div style={{
                backgroundColor: '#e2e8f0',
                borderRadius: '9999px',
                height: 10,
                overflow: 'hidden',
            }}>
                <div style={{
                    height: '100%',
                    width: `${pct}%`,
                    backgroundColor: '#2563eb',
                    borderRadius: '9999px',
                    transition: 'width 0.15s ease',
                }} />
            </div>
            <div style={{
                textAlign: 'right',
                fontSize: '0.75rem',
                color: 'var(--blue_secondary)',
                marginTop: '0.25rem',
            }}>
                {pct}%
            </div>
        </div>
    );
};

export default ProgressBar;
