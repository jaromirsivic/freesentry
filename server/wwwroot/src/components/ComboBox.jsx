import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const ComboBox = ({
    label,
    items = [],
    value,
    onChange,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const selectStyle = {
        padding: '0.5rem',
        borderRadius: '0.375rem',
        border: '1px solid #cbd5e1',
        backgroundColor: disabled ? '#f1f5f9' : '#ffffff',
        cursor: disabled ? 'not-allowed' : 'pointer',
        width: '100%',
        outline: 'none'
    };

    return (
        <div className="custom-combobox responsive-input-container" style={{ width: '100%', opacity: disabled ? 0.5 : 1, gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <label className="combobox-label" style={{ display: labelWidth ? 'inline-block' : 'block', width: labelWidth, minWidth: labelWidth, marginBottom: '0.25rem', whiteSpace: 'nowrap' }}>{label}</label>}
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                disabled={disabled}
                style={selectStyle}
            >
                {items.map((item, index) => (
                    <option
                        key={index}
                        value={item.value}
                        disabled={item.disabled}
                        style={{
                            backgroundColor: item.color || (item.disabled ? '#f1f5f9' : '#ffffff'),
                            textDecoration: item.disabled ? 'line-through' : 'none',
                            color: item.disabled ? '#94a3b8' : 'inherit'
                        }}
                    >
                        {item.label}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default ComboBox;
