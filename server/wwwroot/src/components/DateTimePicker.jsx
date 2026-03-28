import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import calendarIcon from '../assets/icons/calendar.svg';
import { getOutlineStyle } from '../lib/outlineStyle';

/**
 * DateTimePicker component for selecting dates and times with calendar popup.
 * Displays as a ComboBox-style dropdown with date/time input trigger.
 * 
 * @param {string} label - Label text for the input
 * @param {string} timeZone - Time zone for date calculation (default: local)
 * @param {Date} value - Selected date/time value
 * @param {Date} minValue - Minimum selectable date (default: Jan 1, 1900)
 * @param {Date} maxValue - Maximum selectable date (default: Dec 31, 2199)
 * @param {string} format - Date/time format (default: based on precision)
 * @param {string} precision - Selection precision: 'years', 'months', 'days', 'hours', 'minutes', 'seconds'
 * @param {Function} onChange - Callback when date/time changes
 * @param {boolean} disabled - Whether the picker is disabled
 * @param {string} labelWidth - Width of the label
 */
const DateTimePicker = ({
    label,
    labelPosition = 'left',
    timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone,
    value,
    minValue,
    maxValue,
    format,
    precision = 'days',
    onChange,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    // Determine default format based on precision
    const getDefaultFormat = useCallback(() => {
        switch (precision) {
            case 'years': return 'yyyy';
            case 'months': return 'yyyy-mm';
            case 'days': return 'yyyy-mm-dd';
            case 'hours': return 'yyyy-mm-dd HH';
            case 'minutes': return 'yyyy-mm-dd HH:MM';
            case 'seconds': return 'yyyy-mm-dd HH:MM:SS';
            default: return 'yyyy-mm-dd';
        }
    }, [precision]);

    const currentFormat = format ?? getDefaultFormat();

    // Helper to get date parts in specified timezone
    const getDatePartsInTimezone = useCallback((date = new Date()) => {
        const formatter = new Intl.DateTimeFormat('en-US', {
            timeZone,
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
        const parts = formatter.formatToParts(date);
        const getPart = (type) => parts.find(p => p.type === type)?.value || '00';
        
        return {
            year: parseInt(getPart('year')),
            month: parseInt(getPart('month')) - 1,
            day: parseInt(getPart('day')),
            hour: parseInt(getPart('hour') === '24' ? '0' : getPart('hour')),
            minute: parseInt(getPart('minute')),
            second: parseInt(getPart('second'))
        };
    }, [timeZone]);

    // Helper to get midnight in specified timezone
    const getMidnight = useCallback((date = new Date()) => {
        const parts = getDatePartsInTimezone(date);
        return new Date(parts.year, parts.month, parts.day, 0, 0, 0, 0);
    }, [getDatePartsInTimezone]);

    // Get default value based on precision
    const getDefaultValue = useCallback(() => {
        const now = new Date();
        const parts = getDatePartsInTimezone(now);
        
        switch (precision) {
            case 'years':
                return new Date(parts.year, 0, 1, 0, 0, 0, 0);
            case 'months':
                return new Date(parts.year, parts.month, 1, 0, 0, 0, 0);
            case 'days':
                return new Date(parts.year, parts.month, parts.day, 0, 0, 0, 0);
            case 'hours':
                return new Date(parts.year, parts.month, parts.day, parts.hour, 0, 0, 0);
            case 'minutes':
                return new Date(parts.year, parts.month, parts.day, parts.hour, parts.minute, 0, 0);
            case 'seconds':
                return new Date(parts.year, parts.month, parts.day, parts.hour, parts.minute, parts.second, 0);
            default:
                return new Date(parts.year, parts.month, parts.day, 0, 0, 0, 0);
        }
    }, [precision, getDatePartsInTimezone]);

    // Default values
    const defaultValue = getDefaultValue();
    const defaultMinValue = new Date(1900, 0, 1, 0, 0, 0, 0);
    const defaultMaxValue = new Date(2199, 11, 31, 23, 59, 59, 999);

    const currentValue = value ?? defaultValue;
    const currentMinValue = minValue ?? defaultMinValue;
    const currentMaxValue = maxValue ?? defaultMaxValue;

    const [isOpen, setIsOpen] = useState(false);
    const [inputValue, setInputValue] = useState('');
    const [viewYear, setViewYear] = useState(currentValue.getFullYear());
    const [viewMonth, setViewMonth] = useState(currentValue.getMonth());
    const [isSelectingYear, setIsSelectingYear] = useState(false);
    const [isSelectingMonth, setIsSelectingMonth] = useState(false);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, openAbove: false, maxHeight: null });
    
    // Time selection state
    const [selectedHour, setSelectedHour] = useState(currentValue.getHours());
    const [selectedMinute, setSelectedMinute] = useState(currentValue.getMinutes());
    const [selectedSecond, setSelectedSecond] = useState(currentValue.getSeconds());
    
    const containerRef = useRef(null);
    const inputRef = useRef(null);
    const triggerRef = useRef(null);
    const dropdownRef = useRef(null);

    // Check if we need time selection
    const needsTimeSelection = ['hours', 'minutes', 'seconds'].includes(precision);
    const needsDaySelection = ['days', 'hours', 'minutes', 'seconds'].includes(precision);
    const needsMonthSelection = ['months', 'days', 'hours', 'minutes', 'seconds'].includes(precision);

    // Calculate dropdown dimensions based on precision
    const getDropdownHeight = () => {
        if (precision === 'years') return 280;
        if (precision === 'months') return 300;
        if (needsTimeSelection) return 440;
        return 340;
    };

    // Calculate dropdown position when opening
    const updateDropdownPosition = useCallback(() => {
        if (!triggerRef.current) return;
        
        const rect = triggerRef.current.getBoundingClientRect();
        const dropdownHeight = getDropdownHeight();
        const dropdownWidth = 300;
        const spaceBelow = window.innerHeight - rect.bottom;
        const spaceAbove = rect.top;
        const margin = 8; // Minimum margin from viewport edges
        
        // Determine if we should open above or below
        const openAbove = spaceBelow < dropdownHeight && spaceAbove > spaceBelow;
        
        // Calculate available height in chosen direction
        const availableHeight = openAbove ? spaceAbove - margin : spaceBelow - margin;
        
        // Use maxHeight only if content would overflow
        const needsScroll = dropdownHeight > availableHeight;
        const maxHeight = needsScroll ? Math.max(200, availableHeight) : null;
        
        // Calculate top position
        let top;
        if (openAbove) {
            const actualHeight = needsScroll ? maxHeight : dropdownHeight;
            top = rect.top - actualHeight - 4;
            // Ensure it doesn't go above the viewport
            if (top < margin) top = margin;
        } else {
            top = rect.bottom + 4;
        }
        
        let left = rect.left;
        if (left + dropdownWidth > window.innerWidth) {
            left = window.innerWidth - dropdownWidth - margin;
        }
        if (left < margin) left = margin;
        
        setDropdownPosition({
            top,
            left,
            openAbove,
            maxHeight
        });
    }, []);

    // Check if timezone uses DST
    const checkDSTForDate = useCallback((year, month, day) => {
        // Check if there's a DST transition on this day by comparing offsets at different hours
        const dstInfo = {
            hasDSTChange: false,
            missingHour: null,
            duplicateHour: null
        };
        
        let prevOffset = null;
        for (let hour = 0; hour < 24; hour++) {
            const date = new Date(year, month, day, hour, 0, 0, 0);
            const offset = date.getTimezoneOffset();
            
            if (prevOffset !== null && offset !== prevOffset) {
                dstInfo.hasDSTChange = true;
                if (offset < prevOffset) {
                    // Spring forward - an hour is skipped
                    dstInfo.missingHour = hour;
                } else {
                    // Fall back - an hour is repeated
                    dstInfo.duplicateHour = hour - 1;
                }
            }
            prevOffset = offset;
        }
        
        return dstInfo;
    }, []);

    // Get valid hours for the selected date (accounting for DST)
    const getValidHours = useCallback(() => {
        const parts = getDatePartsInTimezone(currentValue);
        const dstInfo = checkDSTForDate(parts.year, parts.month, parts.day);
        
        const hours = [];
        for (let h = 0; h < 24; h++) {
            const isSkipped = dstInfo.missingHour === h;
            const isDuplicate = dstInfo.duplicateHour === h;
            hours.push({ 
                value: h, 
                isSkipped, 
                isDuplicate,
                label: isDuplicate ? `${String(h).padStart(2, '0')} (DST)` : String(h).padStart(2, '0')
            });
        }
        return hours;
    }, [currentValue, getDatePartsInTimezone, checkDSTForDate]);

    // Format date/time according to format string
    const formatDateTime = useCallback((date) => {
        if (!date || isNaN(date.getTime())) return '';
        
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');
        
        return currentFormat
            .replace(/yyyy/gi, year)
            .replace(/mm/g, month)
            .replace(/MM/g, minutes)
            .replace(/dd/gi, day)
            .replace(/HH/g, hours)
            .replace(/SS/g, seconds);
    }, [currentFormat]);

    // Parse date/time from string based on format
    const parseDateTime = useCallback((str) => {
        if (!str || typeof str !== 'string') return null;
        
        // Extract components based on precision
        let year, month = 0, day = 1, hour = 0, minute = 0, second = 0;
        
        const formatLower = currentFormat.toLowerCase();
        
        // Find positions of each component
        const yearMatch = formatLower.match(/yyyy/);
        const monthMatch = currentFormat.match(/mm/);
        const dayMatch = formatLower.match(/dd/);
        const hourMatch = currentFormat.match(/HH/);
        const minuteMatch = currentFormat.match(/MM/);
        const secondMatch = currentFormat.match(/SS/);
        
        // Extract based on positions
        if (yearMatch) {
            const idx = formatLower.indexOf('yyyy');
            year = parseInt(str.substring(idx, idx + 4), 10);
        }
        
        if (monthMatch && needsMonthSelection) {
            const idx = currentFormat.indexOf('mm');
            month = parseInt(str.substring(idx, idx + 2), 10) - 1;
        }
        
        if (dayMatch && needsDaySelection) {
            const idx = formatLower.indexOf('dd');
            day = parseInt(str.substring(idx, idx + 2), 10);
        }
        
        if (hourMatch && ['hours', 'minutes', 'seconds'].includes(precision)) {
            const idx = currentFormat.indexOf('HH');
            hour = parseInt(str.substring(idx, idx + 2), 10);
        }
        
        if (minuteMatch && ['minutes', 'seconds'].includes(precision)) {
            const idx = currentFormat.indexOf('MM');
            minute = parseInt(str.substring(idx, idx + 2), 10);
        }
        
        if (secondMatch && precision === 'seconds') {
            const idx = currentFormat.indexOf('SS');
            second = parseInt(str.substring(idx, idx + 2), 10);
        }
        
        if (isNaN(year)) return null;
        if (needsMonthSelection && (isNaN(month) || month < 0 || month > 11)) return null;
        if (needsDaySelection && (isNaN(day) || day < 1 || day > 31)) return null;
        if (needsTimeSelection && (isNaN(hour) || hour < 0 || hour > 23)) return null;
        if (['minutes', 'seconds'].includes(precision) && (isNaN(minute) || minute < 0 || minute > 59)) return null;
        if (precision === 'seconds' && (isNaN(second) || second < 0 || second > 59)) return null;
        
        const result = new Date(year, month, day, hour, minute, second, 0);
        
        // Verify the date is valid
        if (needsDaySelection && (result.getDate() !== day || result.getMonth() !== month)) {
            return null;
        }
        
        return result;
    }, [currentFormat, precision, needsMonthSelection, needsDaySelection, needsTimeSelection]);

    // Check if date is within bounds
    const isDateInBounds = useCallback((date) => {
        if (!date) return false;
        const time = date.getTime();
        return time >= currentMinValue.getTime() && time <= currentMaxValue.getTime();
    }, [currentMinValue, currentMaxValue]);

    // Update input value when external value changes
    useEffect(() => {
        setInputValue(formatDateTime(currentValue));
        setViewYear(currentValue.getFullYear());
        setViewMonth(currentValue.getMonth());
        setSelectedHour(currentValue.getHours());
        setSelectedMinute(currentValue.getMinutes());
        setSelectedSecond(currentValue.getSeconds());
    }, [currentValue, formatDateTime]);

    // Close dropdown on click outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            const isInsideContainer = containerRef.current && containerRef.current.contains(e.target);
            const isInsideDropdown = dropdownRef.current && dropdownRef.current.contains(e.target);
            
            if (!isInsideContainer && !isInsideDropdown) {
                setIsOpen(false);
                setIsSelectingYear(false);
                setIsSelectingMonth(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Recalculate dropdown position on window resize when open
    useEffect(() => {
        if (!isOpen) return;
        
        const handleResize = () => {
            updateDropdownPosition();
        };
        
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [isOpen, updateDropdownPosition]);

    // Handle input change
    const handleInputChange = (e) => {
        const val = e.target.value;
        setInputValue(val);
        
        const parsed = parseDateTime(val);
        if (parsed && isDateInBounds(parsed)) {
            if (onChange) {
                onChange(parsed);
            }
            setViewYear(parsed.getFullYear());
            setViewMonth(parsed.getMonth());
            setSelectedHour(parsed.getHours());
            setSelectedMinute(parsed.getMinutes());
            setSelectedSecond(parsed.getSeconds());
        }
    };

    // Handle input blur - validate and revert if invalid
    const handleInputBlur = () => {
        const parsed = parseDateTime(inputValue);
        if (!parsed || !isDateInBounds(parsed)) {
            setInputValue(formatDateTime(currentValue));
        }
    };

    // Handle date selection from calendar
    const selectDate = (year, month, day) => {
        const newDate = new Date(
            year, 
            month, 
            day, 
            needsTimeSelection ? selectedHour : 0,
            ['minutes', 'seconds'].includes(precision) ? selectedMinute : 0,
            precision === 'seconds' ? selectedSecond : 0,
            0
        );
        if (isDateInBounds(newDate)) {
            if (onChange) {
                onChange(newDate);
            }
            setInputValue(formatDateTime(newDate));
            if (!needsTimeSelection) {
                setIsOpen(false);
            }
        }
    };

    // Handle year selection (for 'years' precision)
    const selectYear = (year) => {
        const newDate = new Date(year, 0, 1, 0, 0, 0, 0);
        if (isDateInBounds(newDate)) {
            if (onChange) {
                onChange(newDate);
            }
            setInputValue(formatDateTime(newDate));
            setIsOpen(false);
        }
    };

    // Handle month selection (for 'months' precision)
    const selectMonth = (year, month) => {
        const newDate = new Date(year, month, 1, 0, 0, 0, 0);
        if (isDateInBounds(newDate)) {
            if (onChange) {
                onChange(newDate);
            }
            setInputValue(formatDateTime(newDate));
            setIsOpen(false);
        }
    };

    // Handle time selection
    const selectTime = (hour, minute, second) => {
        const newDate = new Date(
            currentValue.getFullYear(),
            currentValue.getMonth(),
            currentValue.getDate(),
            hour,
            minute,
            second,
            0
        );
        if (isDateInBounds(newDate)) {
            if (onChange) {
                onChange(newDate);
            }
            setInputValue(formatDateTime(newDate));
            setSelectedHour(hour);
            setSelectedMinute(minute);
            setSelectedSecond(second);
        }
    };

    // Navigate months
    const prevMonth = () => {
        if (viewMonth === 0) {
            setViewMonth(11);
            setViewYear(viewYear - 1);
        } else {
            setViewMonth(viewMonth - 1);
        }
    };

    const nextMonth = () => {
        if (viewMonth === 11) {
            setViewMonth(0);
            setViewYear(viewYear + 1);
        } else {
            setViewMonth(viewMonth + 1);
        }
    };

    // Generate calendar days
    const generateCalendarDays = () => {
        const firstDayOfMonth = new Date(viewYear, viewMonth, 1);
        const lastDayOfMonth = new Date(viewYear, viewMonth + 1, 0);
        const startDay = firstDayOfMonth.getDay();
        const daysInMonth = lastDayOfMonth.getDate();
        
        const prevMonthLastDay = new Date(viewYear, viewMonth, 0).getDate();
        
        const days = [];
        
        for (let i = startDay - 1; i >= 0; i--) {
            const day = prevMonthLastDay - i;
            const prevM = viewMonth === 0 ? 11 : viewMonth - 1;
            const prevY = viewMonth === 0 ? viewYear - 1 : viewYear;
            days.push({ day, month: prevM, year: prevY, isCurrentMonth: false });
        }
        
        for (let day = 1; day <= daysInMonth; day++) {
            days.push({ day, month: viewMonth, year: viewYear, isCurrentMonth: true });
        }
        
        const remaining = 42 - days.length;
        for (let day = 1; day <= remaining; day++) {
            const nextM = viewMonth === 11 ? 0 : viewMonth + 1;
            const nextY = viewMonth === 11 ? viewYear + 1 : viewYear;
            days.push({ day, month: nextM, year: nextY, isCurrentMonth: false });
        }
        
        return days;
    };

    // Check if a day is selected
    const isSelected = (year, month, day) => {
        return currentValue.getFullYear() === year &&
               currentValue.getMonth() === month &&
               currentValue.getDate() === day;
    };

    // Check if a day is today
    const isToday = (year, month, day) => {
        const today = new Date();
        return today.getFullYear() === year &&
               today.getMonth() === month &&
               today.getDate() === day;
    };

    // Generate years for year picker
    const generateYears = () => {
        const years = [];
        const startYear = Math.max(currentMinValue.getFullYear(), viewYear - 10);
        const endYear = Math.min(currentMaxValue.getFullYear(), viewYear + 10);
        for (let y = startYear; y <= endYear; y++) {
            years.push(y);
        }
        return years;
    };

    // Month names
    const monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];
    const dayNames = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

    const calendarDays = generateCalendarDays();
    const validHours = getValidHours();

    // Today/Now button handler
    const goToNow = () => {
        const now = getDefaultValue();
        if (isDateInBounds(now)) {
            if (onChange) {
                onChange(now);
            }
            setInputValue(formatDateTime(now));
            setViewYear(now.getFullYear());
            setViewMonth(now.getMonth());
            setSelectedHour(now.getHours());
            setSelectedMinute(now.getMinutes());
            setSelectedSecond(now.getSeconds());
            setIsOpen(false);
        }
    };

    // Button style helper
    const buttonStyle = {
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        padding: '4px 8px',
        borderRadius: '4px',
        color: '#374151'
    };

    // Render year-only picker
    const renderYearPicker = () => (
        <div>
            <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '12px' 
            }}>
                <button
                    onClick={() => setViewYear(viewYear - 20)}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ‹‹
                </button>
                <span style={{ fontWeight: '600', fontSize: '1rem', color: '#1f2937' }}>
                    Select Year
                </span>
                <button
                    onClick={() => setViewYear(viewYear + 20)}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ››
                </button>
            </div>
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(4, 1fr)', 
                gap: '4px',
                maxHeight: '240px',
                overflowY: 'auto'
            }}>
                {generateYears().map(y => {
                    const yearDate = new Date(y, 0, 1);
                    const inBounds = y >= currentMinValue.getFullYear() && y <= currentMaxValue.getFullYear();
                    const isSelectedYear = y === currentValue.getFullYear();
                    
                    return (
                        <button
                            key={y}
                            onClick={() => precision === 'years' ? selectYear(y) : (() => {
                                setViewYear(y);
                                setIsSelectingYear(false);
                            })()}
                            disabled={!inBounds}
                            style={{
                                padding: '8px 4px',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: inBounds ? 'pointer' : 'not-allowed',
                                backgroundColor: isSelectedYear ? 'var(--blue_primary)' : 'transparent',
                                color: isSelectedYear ? '#fff' : inBounds ? '#374151' : '#9ca3af',
                                fontWeight: isSelectedYear ? '600' : '400',
                                fontSize: '0.85rem',
                                opacity: inBounds ? 1 : 0.5
                            }}
                            onMouseOver={(e) => { if (inBounds && !isSelectedYear) e.target.style.backgroundColor = '#e5e7eb'; }}
                            onMouseOut={(e) => { if (inBounds && !isSelectedYear) e.target.style.backgroundColor = 'transparent'; }}
                        >
                            {y}
                        </button>
                    );
                })}
            </div>
        </div>
    );

    // Render month-only picker
    const renderMonthPicker = () => (
        <div>
            <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '12px' 
            }}>
                <button
                    onClick={() => setViewYear(viewYear - 1)}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ‹
                </button>
                <button
                    onClick={() => setIsSelectingYear(true)}
                    style={{ ...buttonStyle, fontWeight: '600', fontSize: '1rem', color: '#1f2937' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    {viewYear}
                </button>
                <button
                    onClick={() => setViewYear(viewYear + 1)}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ›
                </button>
            </div>
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(3, 1fr)', 
                gap: '8px'
            }}>
                {monthNames.map((name, idx) => {
                    const monthDate = new Date(viewYear, idx, 1);
                    const inBounds = isDateInBounds(monthDate) || 
                        (monthDate >= new Date(currentMinValue.getFullYear(), currentMinValue.getMonth(), 1) &&
                         monthDate <= new Date(currentMaxValue.getFullYear(), currentMaxValue.getMonth(), 1));
                    const isSelectedMonth = idx === currentValue.getMonth() && viewYear === currentValue.getFullYear();
                    
                    return (
                        <button
                            key={idx}
                            onClick={() => precision === 'months' ? selectMonth(viewYear, idx) : (() => {
                                setViewMonth(idx);
                                setIsSelectingMonth(false);
                            })()}
                            disabled={!inBounds}
                            style={{
                                padding: '10px 4px',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: inBounds ? 'pointer' : 'not-allowed',
                                backgroundColor: isSelectedMonth ? 'var(--blue_primary)' : 'transparent',
                                color: isSelectedMonth ? '#fff' : inBounds ? '#374151' : '#9ca3af',
                                fontWeight: isSelectedMonth ? '600' : '400',
                                fontSize: '0.85rem',
                                opacity: inBounds ? 1 : 0.5
                            }}
                            onMouseOver={(e) => { if (inBounds && !isSelectedMonth) e.target.style.backgroundColor = '#e5e7eb'; }}
                            onMouseOut={(e) => { if (inBounds && !isSelectedMonth) e.target.style.backgroundColor = 'transparent'; }}
                        >
                            {name.slice(0, 3)}
                        </button>
                    );
                })}
            </div>
        </div>
    );

    // Render time picker section
    const renderTimePicker = () => (
        <div style={{ 
            marginTop: '12px', 
            paddingTop: '12px', 
            borderTop: '1px solid #e5e7eb'
        }}>
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                gap: '8px',
                marginBottom: '8px'
            }}>
                <span style={{ fontWeight: '500', fontSize: '0.85rem', color: '#6b7280' }}>Time:</span>
            </div>
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                gap: '4px'
            }}>
                {/* Hours */}
                <select
                    value={selectedHour}
                    onChange={(e) => {
                        const newHour = parseInt(e.target.value, 10);
                        selectTime(
                            newHour, 
                            ['minutes', 'seconds'].includes(precision) ? selectedMinute : 0,
                            precision === 'seconds' ? selectedSecond : 0
                        );
                    }}
                    style={{
                        padding: '6px 8px',
                        border: '1px solid #d1d5db',
                        borderRadius: '4px',
                        fontSize: '0.9rem',
                        fontFamily: 'monospace',
                        cursor: 'pointer',
                        backgroundColor: '#fff'
                    }}
                >
                    {validHours.map(h => (
                        <option 
                            key={h.value} 
                            value={h.value}
                            disabled={h.isSkipped}
                            style={{ color: h.isSkipped ? '#9ca3af' : '#1f2937' }}
                        >
                            {h.label}
                        </option>
                    ))}
                </select>
                
                {/* Minutes */}
                {['minutes', 'seconds'].includes(precision) && (
                    <>
                        <span style={{ fontSize: '1.2rem', fontWeight: '600' }}>:</span>
                        <select
                            value={selectedMinute}
                            onChange={(e) => {
                                const newMinute = parseInt(e.target.value, 10);
                                selectTime(
                                    selectedHour, 
                                    newMinute,
                                    precision === 'seconds' ? selectedSecond : 0
                                );
                            }}
                            style={{
                                padding: '6px 8px',
                                border: '1px solid #d1d5db',
                                borderRadius: '4px',
                                fontSize: '0.9rem',
                                fontFamily: 'monospace',
                                cursor: 'pointer',
                                backgroundColor: '#fff'
                            }}
                        >
                            {Array.from({ length: 60 }, (_, i) => (
                                <option key={i} value={i}>
                                    {String(i).padStart(2, '0')}
                                </option>
                            ))}
                        </select>
                    </>
                )}
                
                {/* Seconds */}
                {precision === 'seconds' && (
                    <>
                        <span style={{ fontSize: '1.2rem', fontWeight: '600' }}>:</span>
                        <select
                            value={selectedSecond}
                            onChange={(e) => {
                                const newSecond = parseInt(e.target.value, 10);
                                selectTime(selectedHour, selectedMinute, newSecond);
                            }}
                            style={{
                                padding: '6px 8px',
                                border: '1px solid #d1d5db',
                                borderRadius: '4px',
                                fontSize: '0.9rem',
                                fontFamily: 'monospace',
                                cursor: 'pointer',
                                backgroundColor: '#fff'
                            }}
                        >
                            {Array.from({ length: 60 }, (_, i) => (
                                <option key={i} value={i}>
                                    {String(i).padStart(2, '0')}
                                </option>
                            ))}
                        </select>
                    </>
                )}
            </div>
            
            {/* DST warning if applicable */}
            {validHours.some(h => h.isSkipped || h.isDuplicate) && (
                <div style={{
                    marginTop: '8px',
                    padding: '6px 8px',
                    backgroundColor: '#fef3c7',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    color: '#92400e',
                    textAlign: 'center'
                }}>
                    ⚠️ DST change on this date
                </div>
            )}
        </div>
    );

    // Render calendar view
    const renderCalendarView = () => (
        <>
            {/* Header with navigation */}
            <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '12px' 
            }}>
                <button
                    onClick={prevMonth}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ‹
                </button>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        onClick={() => setIsSelectingMonth(true)}
                        style={{ ...buttonStyle, fontWeight: '600', fontSize: '1rem', color: '#1f2937' }}
                        onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                        onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                    >
                        {monthNames[viewMonth]}
                    </button>
                    <button
                        onClick={() => setIsSelectingYear(true)}
                        style={{ ...buttonStyle, fontWeight: '600', fontSize: '1rem', color: '#1f2937' }}
                        onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                        onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                    >
                        {viewYear}
                    </button>
                </div>
                <button
                    onClick={nextMonth}
                    style={{ ...buttonStyle, fontSize: '1.2rem' }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#f3f4f6'}
                    onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                    ›
                </button>
            </div>

            {/* Day names header */}
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(7, 1fr)', 
                gap: '2px',
                marginBottom: '4px'
            }}>
                {dayNames.map(d => (
                    <div 
                        key={d} 
                        style={{ 
                            textAlign: 'center', 
                            fontSize: '0.75rem', 
                            color: '#6b7280', 
                            fontWeight: '500',
                            padding: '4px 0'
                        }}
                    >
                        {d}
                    </div>
                ))}
            </div>

            {/* Calendar grid */}
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(7, 1fr)', 
                gap: '2px' 
            }}>
                {calendarDays.map((dayInfo, idx) => {
                    const date = new Date(
                        dayInfo.year, 
                        dayInfo.month, 
                        dayInfo.day,
                        needsTimeSelection ? selectedHour : 0,
                        ['minutes', 'seconds'].includes(precision) ? selectedMinute : 0,
                        precision === 'seconds' ? selectedSecond : 0
                    );
                    const dayDate = new Date(dayInfo.year, dayInfo.month, dayInfo.day, 0, 0, 0, 0);
                    const minDay = new Date(currentMinValue.getFullYear(), currentMinValue.getMonth(), currentMinValue.getDate(), 0, 0, 0, 0);
                    const maxDay = new Date(currentMaxValue.getFullYear(), currentMaxValue.getMonth(), currentMaxValue.getDate(), 0, 0, 0, 0);
                    const inBounds = dayDate >= minDay && dayDate <= maxDay;
                    const selected = isSelected(dayInfo.year, dayInfo.month, dayInfo.day);
                    const todayCheck = isToday(dayInfo.year, dayInfo.month, dayInfo.day);
                    
                    return (
                        <button
                            key={idx}
                            onClick={() => inBounds && selectDate(dayInfo.year, dayInfo.month, dayInfo.day)}
                            disabled={!inBounds}
                            style={{
                                width: '32px',
                                height: '32px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                border: todayCheck && !selected ? '2px solid var(--blue_primary)' : 'none',
                                borderRadius: '50%',
                                cursor: inBounds ? 'pointer' : 'not-allowed',
                                backgroundColor: selected ? 'var(--blue_primary)' : 'transparent',
                                color: selected 
                                    ? '#fff' 
                                    : !inBounds 
                                        ? '#d1d5db'
                                        : dayInfo.isCurrentMonth 
                                            ? '#1f2937' 
                                            : '#9ca3af',
                                fontWeight: todayCheck || selected ? '600' : '400',
                                fontSize: '0.85rem',
                                transition: 'background-color 0.15s',
                                opacity: inBounds ? 1 : 0.5
                            }}
                            onMouseOver={(e) => { 
                                if (inBounds && !selected) {
                                    e.target.style.backgroundColor = '#e5e7eb'; 
                                }
                            }}
                            onMouseOut={(e) => { 
                                if (inBounds && !selected) {
                                    e.target.style.backgroundColor = 'transparent'; 
                                }
                            }}
                        >
                            {dayInfo.day}
                        </button>
                    );
                })}
            </div>

            {/* Time picker for hours/minutes/seconds precision */}
            {needsTimeSelection && renderTimePicker()}
        </>
    );

    // Get the button label based on precision
    const getNowButtonLabel = () => {
        if (needsTimeSelection) return 'Now';
        if (precision === 'months') return 'This Month';
        if (precision === 'years') return 'This Year';
        return 'Today';
    };

    // Determine input width based on precision
    const getInputMinWidth = () => {
        switch (precision) {
            case 'years': return '80px';
            case 'months': return '100px';
            case 'days': return '120px';
            case 'hours': return '140px';
            case 'minutes': return '160px';
            case 'seconds': return '180px';
            default: return '120px';
        }
    };

    return (
        <div 
            className={`custom-date-time-picker responsive-input-container ${labelPosition === 'top' ? 'top-label' : ''}`}
            ref={containerRef} 
            style={{ 
                position: 'relative', 
                opacity: disabled ? 0.5 : 1, 
                pointerEvents: disabled ? 'none' : 'auto', 
                gap: '1rem',
                ...getOutlineStyle(outline),
                ...style
            }}
        >
            {label && (
                <span style={{ 
                    whiteSpace: 'nowrap', 
                    width: labelWidth, 
                    minWidth: labelWidth, 
                    display: labelWidth ? 'inline-block' : 'inline' 
                }}>
                    {label}
                </span>
            )}
            
            {/* Input trigger */}
            <div
                ref={triggerRef}
                onClick={() => {
                    if (disabled) return;
                    if (!isOpen) {
                        updateDropdownPosition();
                    }
                    setIsOpen(!isOpen);
                    setIsSelectingYear(precision === 'years');
                    setIsSelectingMonth(false);
                }}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '6px 10px',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    backgroundColor: '#fff',
                    minWidth: getInputMinWidth()
                }}
            >
                {/* Calendar icon */}
                <img src={calendarIcon} alt="" width="20" height="20" style={{ opacity: 0.7 }} />
                <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={handleInputChange}
                    onBlur={handleInputBlur}
                    onClick={(e) => e.stopPropagation()}
                    disabled={disabled}
                    placeholder={currentFormat}
                    style={{
                        border: 'none',
                        outline: 'none',
                        fontSize: '0.9rem',
                        fontFamily: 'monospace',
                        flex: 1,
                        backgroundColor: 'transparent',
                        minWidth: '100px'
                    }}
                />
                {!disabled && <span style={{ fontSize: '0.7rem', color: '#666' }}>▼</span>}
            </div>

            {/* Calendar dropdown - rendered via portal */}
            {isOpen && createPortal(
                <div
                    ref={dropdownRef}
                    style={{
                        position: 'fixed',
                        top: dropdownPosition.top,
                        left: dropdownPosition.left,
                        width: '300px',
                        padding: '12px',
                        backgroundColor: '#ffffff',
                        borderRadius: '8px',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
                        zIndex: 10000,
                        userSelect: 'none',
                        maxHeight: dropdownPosition.maxHeight ? `${dropdownPosition.maxHeight}px` : undefined,
                        overflowY: dropdownPosition.maxHeight ? 'auto' : undefined,
                        overflowX: 'hidden'
                    }}
                >
                    {/* Year selector view - for 'years' precision or when selecting year */}
                    {(precision === 'years' || isSelectingYear) && renderYearPicker()}

                    {/* Month selector view - for 'months' precision or when selecting month */}
                    {precision === 'months' && !isSelectingYear && renderMonthPicker()}
                    {isSelectingMonth && !isSelectingYear && precision !== 'months' && renderMonthPicker()}

                    {/* Calendar view - for days/hours/minutes/seconds precision */}
                    {needsDaySelection && !isSelectingYear && !isSelectingMonth && renderCalendarView()}

                    {/* Footer with Today/Now button */}
                    <div style={{ 
                        marginTop: '12px', 
                        paddingTop: '12px', 
                        borderTop: '1px solid #e5e7eb',
                        display: 'flex',
                        justifyContent: 'center',
                        gap: '8px'
                    }}>
                        <button
                            onClick={goToNow}
                            style={{
                                padding: '6px 16px',
                                border: '1px solid var(--blue_primary)',
                                borderRadius: '4px',
                                backgroundColor: 'transparent',
                                color: 'var(--blue_primary)',
                                cursor: 'pointer',
                                fontSize: '0.85rem',
                                fontWeight: '500',
                                transition: 'all 0.15s'
                            }}
                            onMouseOver={(e) => { 
                                e.target.style.backgroundColor = 'var(--blue_primary)'; 
                                e.target.style.color = '#fff';
                            }}
                            onMouseOut={(e) => { 
                                e.target.style.backgroundColor = 'transparent'; 
                                e.target.style.color = 'var(--blue_primary)';
                            }}
                        >
                            {getNowButtonLabel()}
                        </button>
                        {needsTimeSelection && (
                            <button
                                onClick={() => setIsOpen(false)}
                                style={{
                                    padding: '6px 16px',
                                    border: '1px solid var(--green_secondary)',
                                    borderRadius: '4px',
                                    backgroundColor: 'var(--green_secondary)',
                                    color: '#fff',
                                    cursor: 'pointer',
                                    fontSize: '0.85rem',
                                    fontWeight: '500',
                                    transition: 'all 0.15s'
                                }}
                                onMouseOver={(e) => { 
                                    e.target.style.backgroundColor = 'var(--green_primary)'; 
                                }}
                                onMouseOut={(e) => { 
                                    e.target.style.backgroundColor = 'var(--green_secondary)'; 
                                }}
                            >
                                Done
                            </button>
                        )}
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
};

export default DateTimePicker;

