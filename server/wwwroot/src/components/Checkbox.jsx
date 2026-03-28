import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const Checkbox = ({
    label,
    labelPosition = 'left',
    value,
    onChange,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const checkboxStyle = {
        width: '20px',
        height: '20px',
        border: `2px solid ${value ? '#3b82f6' : '#cbd5e1'}`,
        backgroundColor: value ? '#3b82f6' : 'transparent',
        borderRadius: '4px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        transition: 'all 0.2s',
        flexShrink: 0
    };

    const containerClass = `responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`;

    return (
        <div className={containerClass} style={{ opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? 'none' : 'auto', gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <span className="checkbox-label" style={{ width: labelWidth, minWidth: labelWidth, display: labelWidth ? 'inline-block' : 'inline' }}>{label}</span>}
            <div
                className="custom-checkbox"
                style={checkboxStyle}
                onClick={() => !disabled && onChange(!value)}
            >
                {value && (
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                        <path d="M2 6L5 9L10 3" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                )}
            </div>
        </div>
    );
};

export default Checkbox;
