import React, { useState, useRef, useId } from 'react';

/**
 * List component — displays a vertical list of text items.
 *
 * Props:
 *   items        {string[]}  Array of text items to display.
 *   itemMaxLines {number}    Max lines per item (default 1000).
 *                            1  → single-line with ellipsis overflow.
 *                            >1 → multi-line clamped with trailing "..." when text overflows.
 *   selectedIndex {number}  Controlled selected index (optional).
 *   onSelect     {function} Called with (index, text) when an item is clicked or navigated to.
 *   style        {object}   Extra styles applied to the outer container.
 */
const List = ({
    items = [],
    itemMaxLines = 1000,
    selectedIndex: controlledIndex,
    onSelect,
    style = {},
}) => {
    const isControlled = controlledIndex !== undefined;
    const [internalIndex, setInternalIndex] = useState(null);
    const [hoveredIndex, setHoveredIndex] = useState(null);
    const containerRef = useRef(null);
    const uid = useId().replace(/:/g, '');
    const clampClass = `list-item-clamped-${uid}`;

    const selectedIndex = isControlled ? controlledIndex : internalIndex;

    const select = (index) => {
        if (!isControlled) setInternalIndex(index);
        onSelect?.(index, items[index]);
    };

    const handleKeyDown = (e) => {
        if (items.length === 0) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            const next = selectedIndex === null ? 0 : Math.min(selectedIndex + 1, items.length - 1);
            select(next);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (selectedIndex === null) return;
            select(Math.max(selectedIndex - 1, 0));
        }
    };

    const itemStyle = (index) => {
        const isSelected = index === selectedIndex;
        const isHovered = index === hoveredIndex && !isSelected;

        const base = {
            padding: '0.4rem 0.75rem',
            cursor: 'pointer',
            userSelect: 'none',
            backgroundColor: isSelected ? 'var(--blue_primary)' : isHovered ? '#f1f5f9' : 'transparent',
            color: isSelected ? '#ffffff' : '#1e293b',
            borderBottom: '1px solid #e2e8f0',
            boxSizing: 'border-box',
            overflowWrap: 'break-word',
            wordBreak: 'break-word',
            transition: 'background-color 0.1s',
        };

        if (itemMaxLines === 1) {
            return {
                ...base,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            };
        }

        // -webkit-line-clamp: WebkitLineClamp is safe as inline style;
        // display + -webkit-box-orient are injected via <style> to survive build tools.
        return {
            ...base,
            WebkitLineClamp: itemMaxLines,
        };
    };

    return (
        <>
            {itemMaxLines !== 1 && (
                <style>{`
                    .${clampClass} {
                        display: -webkit-box !important;
                        -webkit-box-orient: vertical !important;
                        overflow: hidden !important;
                    }
                `}</style>
            )}
        <div
            ref={containerRef}
            tabIndex={0}
            onKeyDown={handleKeyDown}
            style={{
                outline: 'none',
                border: '1px solid #cbd5e1',
                borderRadius: '0.375rem',
                overflowY: 'auto',
                ...style,
            }}
        >
            {items.map((text, index) => (
                <div
                    key={index}
                    className={itemMaxLines !== 1 ? clampClass : undefined}
                    style={itemStyle(index)}
                    onClick={() => {
                        select(index);
                        containerRef.current?.focus();
                    }}
                    onMouseEnter={() => setHoveredIndex(index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                >
                    {text}
                </div>
            ))}
        </div>
        </>
    );
};

export default List;
