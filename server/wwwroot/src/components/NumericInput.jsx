import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const NumericInput = ({
    label,
    labelPosition = 'top',
    value,
    onChange,
    min,
    max,
    step = 1,
    decimalPlaces,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const [localValue, setLocalValue] = React.useState('');
    const initialDelayRef = React.useRef(null);
    const repeatIntervalRef = React.useRef(null);
    const valueRef = React.useRef(value);

    React.useEffect(() => {
        valueRef.current = value;
        if (value !== undefined && value !== null) {
            setLocalValue(decimalPlaces !== undefined ? value.toFixed(decimalPlaces) : String(value));
        } else {
            setLocalValue('');
        }
    }, [value, decimalPlaces]);

    const handleBlur = () => {
        let val = parseFloat(localValue);
        if (isNaN(val)) {
            // Revert to original value if invalid
            setLocalValue(decimalPlaces !== undefined ? value.toFixed(decimalPlaces) : String(value));
            return;
        }
        applyValue(val);
    };

    const applyValue = (val) => {
        // Clamp value
        if (min !== undefined && val < min) val = min;
        if (max !== undefined && val > max) val = max;

        // Apply decimal places if needed to avoid precision issues before sending to onChange
        // but generally we want to keep the number as number.
        // If decimalPlaces is set, we might want to round it.
        if (decimalPlaces !== undefined) {
            val = parseFloat(val.toFixed(decimalPlaces));
        }

        // Notify change
        onChange(val);

        // Format display
        setLocalValue(decimalPlaces !== undefined ? val.toFixed(decimalPlaces) : String(val));
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleBlur();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            handleIncrement();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            handleDecrement();
        }
    };

    const handleChange = (e) => {
        setLocalValue(e.target.value);
    };

    const handleIncrement = () => {
        if (disabled) return;
        let currentVal = parseFloat(localValue);
        if (isNaN(currentVal)) currentVal = value || 0;
        applyValue(currentVal + step);
    };

    const handleDecrement = () => {
        if (disabled) return;
        let currentVal = parseFloat(localValue);
        if (isNaN(currentVal)) currentVal = value || 0;
        applyValue(currentVal - step);
    };

    const stopRepeating = () => {
        if (initialDelayRef.current) {
            clearTimeout(initialDelayRef.current);
            initialDelayRef.current = null;
        }
        if (repeatIntervalRef.current) {
            clearInterval(repeatIntervalRef.current);
            repeatIntervalRef.current = null;
        }
    };

    const handleMouseDown = (direction) => {
        if (disabled) return;
        
        // Execute immediately on click
        if (direction === 'up') {
            handleIncrement();
        } else {
            handleDecrement();
        }

        // Helper to apply value change using ref (avoids stale closure)
        const applyStep = () => {
            let currentVal = valueRef.current;
            if (currentVal === undefined || currentVal === null) currentVal = 0;
            
            const newVal = direction === 'up' ? currentVal + step : currentVal - step;
            
            // Check bounds before applying
            if (direction === 'up' && max !== undefined && newVal > max) {
                stopRepeating();
                return;
            }
            if (direction === 'down' && min !== undefined && newVal < min) {
                stopRepeating();
                return;
            }
            
            applyValue(newVal);
        };

        // Start repeating after 1 second delay
        initialDelayRef.current = setTimeout(() => {
            repeatIntervalRef.current = setInterval(applyStep, 100); // Repeat every 0.1 second
        }, 500); // Initial delay of 0.5 second
    };

    const handleMouseUp = () => {
        stopRepeating();
    };

    const handleMouseLeave = () => {
        stopRepeating();
    };

    // Cleanup timers on unmount
    React.useEffect(() => {
        return () => {
            stopRepeating();
        };
    }, []);

    const containerStyle = {
        display: 'flex',
        flexDirection: labelPosition === 'top' ? 'column' : 'row',
        alignItems: 'center', // Center label and input block
        justifyContent: labelPosition === 'top' ? 'flex-start' : 'space-between',
        width: '100%',
        gap: labelPosition === 'left' ? '1rem' : '0',
        ...getOutlineStyle(outline),
        ...style
    };

    const inputWrapperStyle = {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        width: style.width || '80px' // Default width if not provided in style
    };

    const inputStyle = {
        width: '100%', // Input takes full width of wrapper less the buttons padding potentially, but better to use flex
        flex: 1,
        padding: '0.25rem 1.5rem 0.25rem 0.25rem', // Right padding for space for buttons would be absolute, or use flex structure
        // Let's use a flex structure for input and buttons
        border: '1px solid #cbd5e1',
        borderRadius: '4px',
        backgroundColor: disabled ? '#f1f5f9' : '#ffffff',
        textAlign: 'center',
        height: '32px', // Fixed height to match buttons
        boxSizing: 'border-box'
    };

    // We will use a different approach: Input and a separate div for buttons side-by-side
    // modifying inputStyle to remove right padding and use flex

    const flexWrapperStyle = {
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'stretch',
        border: '1px solid #cbd5e1',
        borderRadius: '4px',
        backgroundColor: disabled ? '#f1f5f9' : '#ffffff',
        overflow: 'hidden', // Contain children
        width: style.width || '100px', // Slightly wider for buttons
        opacity: disabled ? 0.5 : 1
    };

    const innerInputStyle = {
        flex: 1,
        border: 'none',
        padding: '0.25rem',
        textAlign: 'center',
        backgroundColor: 'transparent',
        outline: 'none',
        width: '0', // Allow flex shrink/grow
        color: '#1e293b'
    };

    const spinnerContainerStyle = {
        display: 'flex',
        flexDirection: 'column',
        width: '20px',
        borderLeft: '1px solid #cbd5e1',
        backgroundColor: '#f8fafc'
    };

    const spinnerButtonStyle = {
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: 'none',
        padding: 0,
        margin: 0,
        backgroundColor: 'transparent',
        cursor: disabled ? 'default' : 'pointer',
        fontSize: '0.6rem',
        color: '#64748b'
    };

    const topSpinnerStyle = {
        ...spinnerButtonStyle,
        borderBottom: '1px solid #cbd5e1'
    };

    const bottomSpinnerStyle = {
        ...spinnerButtonStyle
    };

    return (
        <div className={`custom-numeric-input responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`} style={containerStyle}>
            {label && <label className="numeric-input-label" style={{
                display: labelWidth ? 'inline-block' : 'block',
                marginBottom: labelPosition === 'top' ? '0.25rem' : '0',
                whiteSpace: 'nowrap',
                textAlign: 'start',
                width: labelWidth || (labelPosition === 'top' ? '100%' : 'auto'),
                minWidth: labelWidth
            }}>{label}</label>}

            <div style={flexWrapperStyle}>
                <input
                    type="text"
                    value={localValue}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyDown={handleKeyDown}
                    disabled={disabled}
                    style={innerInputStyle}
                />
                <div style={spinnerContainerStyle}>
                    <button
                        type="button" // Prevent form submission
                        style={topSpinnerStyle}
                        onMouseDown={() => handleMouseDown('up')}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseLeave}
                        tabIndex={-1}
                        disabled={disabled}
                    >
                        ▲
                    </button>
                    <button
                        type="button"
                        style={bottomSpinnerStyle}
                        onMouseDown={() => handleMouseDown('down')}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseLeave}
                        tabIndex={-1}
                        disabled={disabled}
                    >
                        ▼
                    </button>
                </div>
            </div>
        </div>
    );
};

export default NumericInput;
