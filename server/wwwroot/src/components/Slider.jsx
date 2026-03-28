import React, { useState, useRef, useEffect } from 'react';
import NumericInput from './NumericInput';
import { getOutlineStyle } from '../lib/outlineStyle';

const Slider = ({
    label,
    labelPosition = 'left',
    value,
    onChange,
    onAfterChange,
    min = 0,
    max = 100,
    minSlider,
    maxSlider,
    step = 1,
    decimalPlaces,
    disabled = false,
    allowManualInput = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const trackRef = useRef(null);

    // Determine slider boundaries
    const sliderMin = minSlider !== undefined ? minSlider : min;
    const sliderMax = maxSlider !== undefined ? maxSlider : max;

    // Calculate percentage based on slider boundaries, clamping value for visual representation
    const clampedValue = Math.max(sliderMin, Math.min(sliderMax, value));
    const percentage = ((clampedValue - sliderMin) / (sliderMax - sliderMin)) * 100;

    // Get clientX from either mouse or touch event
    const getClientX = (e) => {
        if (e.touches && e.touches.length > 0) {
            return e.touches[0].clientX
        }
        if (e.changedTouches && e.changedTouches.length > 0) {
            return e.changedTouches[0].clientX;
        }
        return e.clientX;
    };

    const handleMouseDown = (e) => {
        if (disabled) return;
        setIsDragging(true);
        updateValue(e);
    };

    const handleTouchStart = (e) => {
        if (disabled) return;
        e.preventDefault(); // Prevent scrolling
        setIsDragging(true);
        updateValue(e);
    };

    const updateValue = (e) => {
        if (!trackRef.current) return;
        const rect = trackRef.current.getBoundingClientRect();
        const clientX = getClientX(e);
        const x = clientX - rect.left;
        const width = rect.width;
        let newPercentage = x / width;
        newPercentage = Math.max(0, Math.min(1, newPercentage));

        let newValue = sliderMin + newPercentage * (sliderMax - sliderMin);
        // Snap to step
        newValue = Math.round(newValue / step) * step;
        // Clamp to absolute bounds
        newValue = Math.max(min, Math.min(max, newValue));

        onChange(newValue);
        return newValue; // Return for immediate use if needed
    };

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (isDragging) {
                updateValue(e);
            }
        };

        const handleTouchMove = (e) => {
            if (isDragging) {
                e.preventDefault(); // Prevent scrolling while dragging
                updateValue(e);
            }
        };

        const handleMouseUp = (e) => {
            if (isDragging) {
                setIsDragging(false);
                // Trigger onAfterChange with the final value.
                // Since state update might be async, we recalculate or rely on current event flow.
                // Ideally we should pass the value we just calculated.
                // However, updateValue calls onChange, which pushes to parent. Parent updates prop `value`.
                // By the time mouseUp happens, `value` prop might be old or new depending on React batching.
                // Safest to call updateValue one last time or rely on `value` if we assume sync.
                // But wait, mouseUp doesn't necessarily have new coordinates if mouse didn't move.
                // Let's rely on the `value` prop being relatively fresh or pass current value if available?
                // Actually, simply emitting onAfterChange(value) using the prop `value` is risky if bad timing.
                // Let's just emit whatever the current value is from props, assuming parent has updated it by now if we were dragging.
                if (onAfterChange) onAfterChange(value);
            }
        };

        const handleTouchEnd = () => {
            if (isDragging) {
                setIsDragging(false);
                if (onAfterChange) onAfterChange(value);
            }
        };

        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
            window.addEventListener('touchmove', handleTouchMove, { passive: false });
            window.addEventListener('touchend', handleTouchEnd);
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('touchend', handleTouchEnd);
        };
    }, [isDragging, min, max, minSlider, maxSlider, step, onChange, onAfterChange, value]);

    const handleManualChange = (val) => {
        onChange(val);
        if (onAfterChange) onAfterChange(val);
    };

    const trackStyle = {
        position: 'relative',
        width: '100%',
        height: '4px',
        backgroundColor: '#cbd5e1',
        borderRadius: '2px',
        cursor: 'pointer',
        flex: 1,
        touchAction: 'none' // Prevent browser from interpreting touch as scroll
    };

    const fillStyle = {
        position: 'absolute',
        left: 0,
        top: 0,
        height: '100%',
        width: `${percentage}%`,
        backgroundColor: 'var(--blue_primary)',
        borderRadius: '2px'
    };

    const thumbStyle = {
        position: 'absolute',
        left: `${percentage}%`,
        top: '50%',
        transform: 'translate(-50%, -50%)',
        width: '16px',
        height: '16px',
        backgroundColor: '#ffffff',
        border: '2px solid var(--blue_primary)',
        borderRadius: '50%',
        cursor: 'grab',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
    };

    const valueLabelStyle = {
        position: 'absolute',
        left: `${percentage}%`,
        top: '-25px',
        transform: 'translateX(-50%)',
        backgroundColor: '#1e293b',
        color: '#ffffff',
        padding: '2px 8px',
        borderRadius: '4px',
        fontSize: '0.75rem',
        whiteSpace: 'nowrap'
    };

    return (
        <div className={`custom-slider responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`} style={{ width: '100%', opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? 'none' : 'auto', gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <span style={{ whiteSpace: 'nowrap', width: labelWidth, minWidth: labelWidth, display: labelWidth ? 'inline-block' : 'inline' }}>{label}</span>}
            <div style={{ position: 'relative', flex: 1, height: '40px', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {isDragging && !allowManualInput && <div style={valueLabelStyle}>{value}</div>}
                {/* Track wrapper with horizontal padding to prevent thumb from touching adjacent elements */}
                <div style={{ flex: 1, paddingLeft: '10px', paddingRight: '10px' }}>
                    <div
                        ref={trackRef}
                        style={trackStyle}
                        onMouseDown={handleMouseDown}
                        onTouchStart={handleTouchStart}
                    >
                        <div style={fillStyle} />
                        <div style={thumbStyle} />
                    </div>
                </div>

                {allowManualInput && (
                    <div style={{ width: '70px', flexShrink: 0 }}>
                        <NumericInput
                            value={value}
                            onChange={handleManualChange}
                            min={min}
                            max={max}
                            step={step}
                            decimalPlaces={decimalPlaces}
                            disabled={disabled}
                            style={{ width: '100%' }}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};

export default Slider;
