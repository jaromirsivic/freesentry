import React, { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Joystick1D component - A 1D joystick/fader control similar to audio mixer sliders.
 * The button snaps back to origin when released with smooth animation.
 * 
 * @param {string} orientation - 'vertical' or 'horizontal' (default: 'vertical')
 * @param {string} mode - 'joystick' (snaps back to origin) or 'slider' (stays where released) (default: 'joystick')
 * @param {string} backgroundColor - Background color of the track area (default: 'transparent')
 * @param {string} rulerColor - Color of the ruler lines and text (default: '#3b82f6')
 * @param {boolean} rulerShowText - Whether to show text labels on ruler (default: true)
 * @param {boolean} rulerVisible - Whether to show ruler lines and text (default: true)
 * @param {number} rulerLineDistance - Distance between major ruler lines in value units (default: 0.2)
 * @param {number} rulerLineWidth - Width/thickness of ruler lines in pixels (default: 1)
 * @param {string} buttonColor - Color of the button background (default: '#4b5563')
 * @param {string} buttonOutline - Color of button outline (default: '#2563eb')
 * @param {string} buttonGroovingColor - Color of the button groovings (default: '#f9fafb')
 * @param {string} socketColor - Color of the socket/rail behind the button (default: '#374151')
 * @param {number} valueOrigin - The rest position value where button snaps to (default: 0)
 * @param {number} value - Current value, rounded to 6 decimal places
 * @param {number} minValue - Minimum value (default: -1)
 * @param {number} maxValue - Maximum value (default: 1)
 * @param {number} snapAnimationDuration - Duration of snap animation in seconds (default: 0.1)
 * @param {number} width - Width of the component in pixels (default: 60 for vertical, 300 for horizontal)
 * @param {number} height - Height of the component in pixels (default: 300 for vertical, 60 for horizontal)
 * @param {Function} onStart - Called when user starts dragging the button
 * @param {Function} onChange - Called when value changes during drag (receives { value })
 * @param {Function} onEnd - Called when user releases the button
 */
const Joystick1D = ({
    orientation = 'vertical',
    mode = 'joystick',
    backgroundColor = 'transparent',
    rulerColor = '#3b82f6',
    rulerShowText = true,
    rulerVisible = true,
    rulerLineDistance = 0.2,
    rulerLineWidth = 1,
    buttonColor = '#4b5563',
    buttonOutline = '#2563eb',
    buttonGroovingColor = '#f9fafb',
    socketColor = '#374151',
    valueOrigin = 0,
    value: controlledValue,
    minValue = -1,
    maxValue = 1,
    snapAnimationDuration = 0.1,
    width,
    height,
    onStart,
    onChange,
    onEnd
}) => {
    const isVertical = orientation === 'vertical';
    
    // Default dimensions based on orientation
    const defaultWidth = isVertical ? 80 : 300;
    const defaultHeight = isVertical ? 300 : 80;
    const actualWidth = width ?? defaultWidth;
    const actualHeight = height ?? defaultHeight;
    
    // Button dimensions
    const buttonWidth = isVertical ? 44 : 60;
    const buttonHeight = isVertical ? 60 : 44;
    
    // Track dimensions (the center rail)
    const trackWidth = isVertical ? 8 : (actualWidth - 40);
    const trackHeight = isVertical ? (actualHeight - 40) : 8;
    
    // State
    const [internalValue, setInternalValue] = useState(valueOrigin);
    const [isDragging, setIsDragging] = useState(false);
    const [isAnimating, setIsAnimating] = useState(false);
    
    const containerRef = useRef(null);
    const animationRef = useRef(null);
    const dragStartRef = useRef({ value: 0, clientPos: 0 });
    
    // Use controlled value if provided, otherwise internal
    const currentValue = controlledValue !== undefined ? controlledValue : internalValue;
    
    // Round value to 6 decimal places
    const roundValue = useCallback((val) => {
        return Math.round(val * 1000000) / 1000000;
    }, []);
    
    // Clamp value within bounds
    const clampValue = useCallback((val) => {
        return Math.min(maxValue, Math.max(minValue, val));
    }, [minValue, maxValue]);
    
    // Convert value to pixel position
    const valueToPixel = useCallback((val) => {
        const range = maxValue - minValue;
        const normalizedValue = (val - minValue) / range;
        
        if (isVertical) {
            // Vertical: top is maxValue, bottom is minValue
            const trackLength = trackHeight - buttonHeight;
            return (1 - normalizedValue) * trackLength;
        } else {
            // Horizontal: left is minValue, right is maxValue
            const trackLength = trackWidth - buttonWidth;
            return normalizedValue * trackLength;
        }
    }, [isVertical, minValue, maxValue, trackHeight, trackWidth, buttonHeight, buttonWidth]);
    
    // Convert pixel position to value
    const pixelToValue = useCallback((pixel) => {
        const range = maxValue - minValue;
        
        if (isVertical) {
            const trackLength = trackHeight - buttonHeight;
            const normalizedValue = 1 - (pixel / trackLength);
            return minValue + normalizedValue * range;
        } else {
            const trackLength = trackWidth - buttonWidth;
            const normalizedValue = pixel / trackLength;
            return minValue + normalizedValue * range;
        }
    }, [isVertical, minValue, maxValue, trackHeight, trackWidth, buttonHeight, buttonWidth]);
    
    // Handle drag start
    const handleDragStart = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        
        // Cancel any ongoing animation
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
        setIsAnimating(false);
        
        const clientPos = e.type.includes('touch') 
            ? (isVertical ? e.touches[0].clientY : e.touches[0].clientX)
            : (isVertical ? e.clientY : e.clientX);
        
        dragStartRef.current = {
            value: currentValue,
            clientPos
        };
        
        setIsDragging(true);
        onStart?.();
    }, [currentValue, isVertical, onStart]);
    
    // Handle drag move
    const handleDragMove = useCallback((e) => {
        if (!isDragging) return;
        
        const clientPos = e.type.includes('touch')
            ? (isVertical ? e.touches[0].clientY : e.touches[0].clientX)
            : (isVertical ? e.clientY : e.clientX);
        
        const delta = clientPos - dragStartRef.current.clientPos;
        
        // Calculate pixel movement based on container
        const trackLength = isVertical ? (trackHeight - buttonHeight) : (trackWidth - buttonWidth);
        const valueRange = maxValue - minValue;
        const valuePerPixel = valueRange / trackLength;
        
        // For vertical, moving down decreases value; for horizontal, moving right increases
        const valueDelta = isVertical ? -delta * valuePerPixel : delta * valuePerPixel;
        const newValue = roundValue(clampValue(dragStartRef.current.value + valueDelta));
        
        if (newValue !== currentValue) {
            setInternalValue(newValue);
            onChange?.({ value: newValue });
        }
    }, [isDragging, isVertical, trackHeight, trackWidth, buttonHeight, buttonWidth, minValue, maxValue, currentValue, roundValue, clampValue, onChange]);
    
    // Handle drag end
    const handleDragEnd = useCallback(() => {
        if (!isDragging) return;
        
        setIsDragging(false);
        
        // In slider mode: button stays where user left it, no snap back
        if (mode === 'slider') {
            // Final onChange with current position, then onEnd
            onChange?.({ value: currentValue });
            onEnd?.();
            return;
        }
        
        // Joystick mode: snap back to valueOrigin
        const startValue = currentValue;
        const targetValue = valueOrigin;
        
        // Immediately report the final value (valueOrigin)
        onChange?.({ value: targetValue });
        onEnd?.();
        
        // If already at target, no animation needed
        if (startValue === targetValue) {
            setInternalValue(targetValue);
            return;
        }
        
        // Visual-only animation back to valueOrigin (no more onChange events)
        const startTime = performance.now();
        const duration = snapAnimationDuration * 1000; // Convert to ms
        
        setIsAnimating(true);
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const animatedValue = roundValue(startValue + (targetValue - startValue) * easeOut);
            
            // Only update visual position, don't trigger onChange
            setInternalValue(animatedValue);
            
            if (progress < 1) {
                animationRef.current = requestAnimationFrame(animate);
            } else {
                setIsAnimating(false);
                setInternalValue(targetValue);
            }
        };
        
        animationRef.current = requestAnimationFrame(animate);
    }, [isDragging, currentValue, valueOrigin, snapAnimationDuration, roundValue, onChange, onEnd, mode]);
    
    // Global event listeners for drag
    useEffect(() => {
        if (isDragging) {
            const handleMove = (e) => handleDragMove(e);
            const handleUp = () => handleDragEnd();
            
            window.addEventListener('mousemove', handleMove);
            window.addEventListener('mouseup', handleUp);
            window.addEventListener('touchmove', handleMove, { passive: false });
            window.addEventListener('touchend', handleUp);
            window.addEventListener('touchcancel', handleUp);
            
            return () => {
                window.removeEventListener('mousemove', handleMove);
                window.removeEventListener('mouseup', handleUp);
                window.removeEventListener('touchmove', handleMove);
                window.removeEventListener('touchend', handleUp);
                window.removeEventListener('touchcancel', handleUp);
            };
        }
    }, [isDragging, handleDragMove, handleDragEnd]);
    
    // Cleanup animation on unmount
    useEffect(() => {
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, []);
    
    // Generate ruler marks
    const generateRulerMarks = () => {
        const marks = [];
        const range = maxValue - minValue;
        const step = rulerLineDistance;
        
        for (let val = minValue; val <= maxValue + 0.0001; val += step) {
            const roundedVal = roundValue(val);
            const isMajor = Math.abs(roundedVal % (step * 5)) < 0.0001 || 
                           Math.abs(roundedVal) < 0.0001 ||
                           Math.abs(roundedVal - minValue) < 0.0001 ||
                           Math.abs(roundedVal - maxValue) < 0.0001;
            
            const position = valueToPixel(roundedVal);
            
            marks.push({
                value: roundedVal,
                position,
                isMajor
            });
        }
        
        return marks;
    };
    
    const rulerMarks = generateRulerMarks();
    const buttonPosition = valueToPixel(currentValue);
    
    // Styles
    const containerStyle = {
        position: 'relative',
        width: actualWidth,
        height: actualHeight,
        backgroundColor,
        userSelect: 'none',
        touchAction: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    };
    
    // Track area style (where the button moves)
    const trackAreaStyle = {
        position: 'absolute',
        zIndex: 2, // Button should be above ruler
        ...(isVertical ? {
            top: 20,
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 50,
        } : {
            left: 20,
            right: 20,
            top: '50%',
            transform: 'translateY(-50%)',
            height: 50,
        })
    };
    
    // Center rail style (socket behind the button)
    const railStyle = {
        position: 'absolute',
        backgroundColor: socketColor,
        borderRadius: 4,
        ...(isVertical ? {
            left: '50%',
            transform: 'translateX(-50%)',
            width: 8,
            top: 0,
            bottom: 0,
        } : {
            top: '50%',
            transform: 'translateY(-50%)',
            height: 8,
            left: 0,
            right: 0,
        })
    };
    
    // Button style
    const buttonStyle = {
        position: 'absolute',
        width: buttonWidth,
        height: buttonHeight,
        backgroundColor: buttonColor,
        border: `2px solid ${buttonOutline}`,
        borderRadius: 4,
        cursor: isDragging ? 'grabbing' : 'grab',
        boxShadow: isDragging 
            ? '2px 2px 8px rgba(0,0,0,0.4)' 
            : '3px 3px 6px rgba(0,0,0,0.3)',
        transition: isAnimating ? 'none' : 'box-shadow 0.2s',
        display: 'flex',
        // Vertical: grooves are horizontal, stacked vertically (flexDirection: column)
        // Horizontal: grooves are vertical, stacked horizontally (flexDirection: row)
        flexDirection: isVertical ? 'column' : 'row',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 2,
        overflow: 'hidden',
        ...(isVertical ? {
            left: '50%',
            transform: 'translateX(-50%)',
            top: buttonPosition,
        } : {
            top: '50%',
            transform: 'translateY(-50%)',
            left: buttonPosition,
        })
    };
    
    // Groove lines on button - perpendicular to movement direction
    // Vertical joystick: horizontal grooves, Horizontal joystick: vertical grooves
    const grooveLines = Array.from({ length: 7 }, (_, i) => {
        const isCenter = i === 3;
        const isOuter = i === 0 || i === 6;
        return (
            <div
                key={i}
                style={{
                    // For vertical orientation: horizontal lines (width > height)
                    // For horizontal orientation: vertical lines (height > width)
                    width: isVertical ? '70%' : 3,
                    height: isVertical ? 3 : '70%',
                    //backgroundColor: isCenter ? '#f9fafb' : isOuter ? '#6b7280' : '#9ca3af',
                    backgroundColor: buttonGroovingColor,
                    borderRadius: 1,
                    opacity: isCenter ? 1 : isOuter ? 0.5 : 0.8
                }}
            />
        );
    });
    
    // Ruler style
    const rulerStyle = {
        position: 'absolute',
        zIndex: 1, // Ruler should be behind button
        ...(isVertical ? {
            right: 8,
            top: 20,
            bottom: 20,
            width: 30,
        } : {
            bottom: 8,
            left: 20,
            right: 20,
            height: 30,
        })
    };
    
    return (
        <div ref={containerRef} style={containerStyle}>
            {/* Track area */}
            <div style={trackAreaStyle}>
                {/* Center rail */}
                <div style={railStyle} />
                
                {/* Draggable button */}
                <div
                    style={buttonStyle}
                    onMouseDown={handleDragStart}
                    onTouchStart={handleDragStart}
                >
                    {grooveLines}
                </div>
            </div>
            
            {/* Ruler */}
            {rulerVisible && (
                <div style={rulerStyle}>
                    {rulerMarks.map((mark, idx) => (
                        <div
                            key={idx}
                            style={{
                                position: 'absolute',
                                ...(isVertical ? {
                                    left: 0,
                                    top: mark.position + (buttonHeight / 2),
                                    height: rulerLineWidth,
                                } : {
                                    top: 0,
                                    left: mark.position + (buttonWidth / 2),
                                    width: rulerLineWidth,
                                })
                            }}
                        >
                            {/* Tick mark */}
                            <div
                                style={{
                                    backgroundColor: rulerColor,
                                    opacity: mark.isMajor ? 1 : 0.5,
                                    ...(isVertical ? {
                                        width: mark.isMajor ? 12 : 6,
                                        height: rulerLineWidth,
                                    } : {
                                        width: rulerLineWidth,
                                        height: mark.isMajor ? 12 : 6,
                                    })
                                }}
                            />
                            {/* Value label - positioned to not affect line placement */}
                            {rulerShowText && mark.isMajor && (
                                <span
                                    style={{
                                        position: 'absolute',
                                        fontSize: '0.65rem',
                                        color: rulerColor,
                                        fontFamily: 'monospace',
                                        fontWeight: 500,
                                        whiteSpace: 'nowrap',
                                        ...(isVertical ? {
                                            // Right of the line for vertical orientation
                                            left: 14,
                                            top: '50%',
                                            transform: 'translateY(-50%)',
                                        } : {
                                            // Below the line for horizontal orientation
                                            top: 14,
                                            left: '50%',
                                            transform: 'translateX(-50%)',
                                        })
                                    }}
                                >
                                    {mark.value.toFixed(1)}
                                </span>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Joystick1D;

