import React, { useState } from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const Textbox = ({
    label,
    labelPosition = 'left',
    value,
    onChange,
    hint,
    disabled = false,
    labelWidth,
    maxLength,
    style = {},
    outline = null
}) => {
    const [isFocused, setIsFocused] = useState(false);

    const inputStyle = {
        padding: '0.5rem',
        borderRadius: '0.375rem',
        border: '1px solid #cbd5e1',
        outline: 'none',
        width: '100%',
        transition: 'border-color 0.2s',
        backgroundColor: disabled ? '#f1f5f9' : '#ffffff',
        borderColor: isFocused ? '#3b82f6' : '#cbd5e1'
    };

    const containerClass = `responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`;

    return (
        <div className={containerClass} style={{ opacity: disabled ? 0.5 : 1, width: '100%', gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <label className="textbox-label" style={{ whiteSpace: 'nowrap', width: labelWidth, minWidth: labelWidth, display: labelWidth ? 'inline-block' : 'inline' }}>{label}</label>}
            <input
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                maxLength={maxLength}
                placeholder={isFocused ? '' : hint}
                disabled={disabled}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                style={inputStyle}
                className="custom-textbox"
            />
        </div>
    );
};

export default Textbox;
