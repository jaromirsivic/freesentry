import React from 'react';

const FEEDBACK_STYLES = {
    error: {
        color: '#ef4444',
        padding: '0.75rem',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderRadius: '4px',
        fontSize: '0.875rem',
        lineHeight: 1.4
    },
    warning: {
        color: '#92400e',
        padding: '0.75rem',
        backgroundColor: 'rgba(245, 158, 11, 0.12)',
        borderRadius: '4px',
        fontSize: '0.875rem',
        lineHeight: 1.4
    }
};

const OperationFeedback = ({ feedback }) => {
    if (!feedback?.message) {
        return null;
    }

    const tone = feedback.tone === 'warning' ? 'warning' : 'error';

    return (
        <div
            role={tone === 'error' ? 'alert' : 'status'}
            aria-live={tone === 'error' ? 'assertive' : 'polite'}
            style={FEEDBACK_STYLES[tone]}
        >
            {feedback.message}
        </div>
    );
};

export default OperationFeedback;
