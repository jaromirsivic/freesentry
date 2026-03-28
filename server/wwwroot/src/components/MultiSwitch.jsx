import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const MultiSwitch = ({
    options = [],
    value,
    onChange,
    orientation = 'horizontal',
    disabled = false,
    selectedColor = '#888888',
    unselectedColor = '#cccccc',
    textColor = '#333333',
    selectedTextColor = '#333333',
    borderRadius = '0.375rem',
    forceButtonsSameWidth = true,
    style = {},
    outline = null
}) => {
    const isHorizontal = orientation === 'horizontal';
    const useGrid = forceButtonsSameWidth && isHorizontal;

    const containerStyle = {
        display: useGrid ? 'grid' : 'flex',
        ...(useGrid
            ? { gridAutoFlow: 'column', gridAutoColumns: '1fr' }
            : { flexDirection: isHorizontal ? 'row' : 'column' }
        ),
        width: 'fit-content',
        borderRadius: borderRadius,
        overflow: 'hidden',
        border: '1px solid #999999',
        opacity: disabled ? 0.5 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
        ...getOutlineStyle(outline),
        ...style
    };

    const getOptionStyle = (option, index) => {
        const isSelected = option.value === value;
        const isLast = index === options.length - 1;

        return {
            backgroundColor: isSelected ? selectedColor : unselectedColor,
            color: isSelected ? selectedTextColor : textColor,
            padding: '0.5rem 0.5rem',
            border: 'none',
            cursor: disabled ? 'not-allowed' : 'pointer',
            fontWeight: isSelected ? 600 : 400,
            transition: 'background-color 0.2s, font-weight 0.2s',
            borderRight: isHorizontal && !isLast ? '1px solid #999999' : 'none',
            borderBottom: !isHorizontal && !isLast ? '1px solid #999999' : 'none',
            ...(!useGrid && { flex: isHorizontal ? 1 : 'none', minWidth: 'max-content' }),
            textAlign: 'center',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
        };
    };

    const handleClick = (option) => {
        if (!disabled && !option.disabled && onChange) {
            onChange(option.value);
        }
    };

    return (
        <div className="multi-switch-container" style={containerStyle}>
            {options.map((option, index) => (
                <button
                    key={option.value}
                    style={{
                        ...getOptionStyle(option, index),
                        opacity: option.disabled ? 0.5 : 1,
                        cursor: option.disabled ? 'not-allowed' : 'pointer'
                    }}
                    onClick={() => handleClick(option)}
                    disabled={disabled || option.disabled}
                    className="multi-switch-option"
                >
                    {option.label}
                </button>
            ))}
        </div>
    );
};

export default MultiSwitch;
