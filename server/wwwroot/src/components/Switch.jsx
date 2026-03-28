import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const Switch = ({
    label,
    labelPosition = 'left',
    value,
    onChange,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const switchStyle = {
        position: 'relative',
        width: '40px',
        height: '24px',
        backgroundColor: value ? '#009900ff' : '#ef4444',
        borderRadius: '24px',
        transition: 'background-color 0.2s',
        cursor: 'pointer',
        flexShrink: 0
    };

    const knobStyle = {
        position: 'absolute',
        top: '2px',
        left: value ? '18px' : '2px',
        width: '20px',
        height: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '50%',
        transition: 'left 0.2s',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
    };

    const containerClass = `responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`;

    return (
        <div className={containerClass} style={{ opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? 'none' : 'auto', gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <span className="switch-label" style={{ width: labelWidth, minWidth: labelWidth, display: labelWidth ? 'inline-block' : 'inline' }}>{label}</span>}
            <div
                className="custom-switch"
                style={switchStyle}
                onClick={() => !disabled && onChange(!value)}
            >
                <div style={knobStyle} />
            </div>
        </div>
    );
};

export default Switch;
