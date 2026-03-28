import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';

/**
 * Universal popup / context-menu component.
 *
 * ## Positioning modes
 *   anchor   – DOM element (or ref) that triggers the menu (button-anchored menus).
 *              The menu opens below/above the element and left/right-aligned,
 *              always choosing the direction with the most available space.
 *   position – { x, y } mouse coordinates (context menus opened on right-click).
 *              The menu opens to the right/left and below/above the cursor,
 *              choosing the direction with the most available space.
 *
 * If the menu is too tall to fit in the chosen direction the component caps its
 * height to the available space and enables a vertical scrollbar.
 *
 * ## Item shapes
 *   { type: 'separator' }
 *     – Horizontal divider line.
 *
 *   { type: 'header', label, icon?, color?, background? }
 *     – Non-clickable group label (e.g. for navigation groups in MenuBar).
 *
 *   { label, icon?, shortcut?, onClick?, disabled?, color?, background?, indent?, allowOpenInNewTab?, href? }
 *     – Regular interactive item.
 *       label            – item text
 *       icon             – any ReactNode shown to the left of the label
 *       shortcut         – keyboard shortcut hint shown on the right (e.g. 'Ctrl+C')
 *       onClick          – called when item is clicked (menu closes automatically)
 *       disabled         – grayed-out, not clickable
 *       color            – CSS color for the item's font
 *       background       – CSS color for the item's background (e.g. to highlight active item)
 *       indent           – adds extra left padding (useful for sub-items)
 *       allowOpenInNewTab – when true, Ctrl+click (or Cmd+click on Mac) opens `href` in a new tab
 *       href             – URL/path used when opening in a new tab (requires allowOpenInNewTab)
 *
 * ## Props
 *   items           – array of item descriptors (see above)
 *   anchor          – DOM element / ref.current used as anchor
 *   position        – { x, y } for context-menu mode
 *   onClose         – called whenever the menu should close
 *   backgroundColor – CSS color of the menu panel (default '#fff', supports rgba)
 *   hoverBackground – background color of a hovered item (default '#f1f5f9')
 *   borderColor     – border / separator color (default '#e2e8f0')
 *   minWidth        – minimum menu width in px (default 160)
 */
const PopupMenu = ({
    items = [],
    anchor = null,
    position = null,
    onClose,
    backgroundColor = '#fff',
    hoverBackground = '#f1f5f9',
    borderColor = '#e2e8f0',
    minWidth = 160,
}) => {
    const menuRef = useRef(null);
    // pos === null means "not yet measured – render invisible to measure"
    const [pos, setPos] = useState(null);

    // ── Close on outside mousedown or Escape ─────────────────────────────────
    useEffect(() => {
        // For anchor-based menus: if the user clicks the anchor element itself,
        // do NOT close here – the anchor's onClick will toggle the menu state.
        const anchorEl = anchor && typeof anchor.contains === 'function' ? anchor : null;

        const handleMouseDown = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) {
                if (anchorEl && anchorEl.contains(e.target)) return;
                onClose();
            }
        };
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') onClose();
        };
        document.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('mousedown', handleMouseDown);
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [onClose, anchor]);

    // ── Smart positioning ─────────────────────────────────────────────────────
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;

        const { width: measuredW, height: measuredH } = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const GAP = 6; // minimum gap from viewport edges
        const w = Math.max(measuredW, minWidth);

        let top, left, maxHeight;

        if (anchor) {
            const rect = typeof anchor.getBoundingClientRect === 'function'
                ? anchor.getBoundingClientRect()
                : anchor;

            // Vertical: prefer opening below the anchor
            const spaceBelow = vh - rect.bottom - GAP;
            const spaceAbove = rect.top - GAP;

            if (spaceBelow >= measuredH || spaceBelow >= spaceAbove) {
                top = rect.bottom + GAP;
                maxHeight = spaceBelow < measuredH ? spaceBelow : undefined;
            } else {
                const availableAbove = Math.min(measuredH, spaceAbove);
                top = rect.top - availableAbove - GAP;
                maxHeight = spaceAbove < measuredH ? spaceAbove : undefined;
            }

            // Horizontal: prefer left-aligned with anchor; fall back to right-aligned
            if (rect.left + w <= vw - GAP) {
                left = rect.left;
            } else {
                left = rect.right - w;
            }
        } else if (position) {
            const { x, y } = position;

            // Vertical: prefer opening below cursor
            const spaceBelow = vh - y - GAP;
            const spaceAbove = y - GAP;

            if (spaceBelow >= measuredH || spaceBelow >= spaceAbove) {
                top = y;
                maxHeight = spaceBelow < measuredH ? spaceBelow : undefined;
            } else {
                const availableAbove = Math.min(measuredH, spaceAbove);
                top = y - availableAbove;
                maxHeight = spaceAbove < measuredH ? spaceAbove : undefined;
            }

            // Horizontal: prefer opening to the right of cursor
            const spaceRight = vw - x - GAP;
            if (spaceRight >= w || spaceRight >= x - GAP) {
                left = x;
            } else {
                left = x - w;
            }
        } else {
            return;
        }

        // Clamp to viewport
        left = Math.max(GAP, Math.min(left, vw - GAP));
        top  = Math.max(GAP, Math.min(top,  vh - GAP));

        setPos({ top, left, maxHeight });
    }, [anchor, position, minWidth]); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Styles ────────────────────────────────────────────────────────────────
    const menuStyle = {
        position: 'fixed',
        top: pos ? pos.top : 0,
        left: pos ? pos.left : 0,
        visibility: pos ? 'visible' : 'hidden',
        zIndex: 99999,
        backgroundColor,
        border: `1px solid ${borderColor}`,
        borderRadius: '0.375rem',
        boxShadow: '0 4px 20px rgba(0,0,0,0.18)',
        minWidth: `${minWidth}px`,
        padding: '0.25rem 0',
        fontSize: '0.875rem',
        boxSizing: 'border-box',
        ...(pos?.maxHeight != null ? { maxHeight: pos.maxHeight, overflowY: 'auto' } : {}),
    };

    // ── Render ────────────────────────────────────────────────────────────────
    const renderItem = (item, index) => {
        if (item.type === 'separator') {
            return (
                <div
                    key={index}
                    style={{ borderTop: `1px solid ${borderColor}`, margin: '0.25rem 0' }}
                />
            );
        }

        if (item.type === 'header') {
            return (
                <div
                    key={index}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.625rem',
                        padding: '0.5rem 0.75rem',
                        cursor: 'default',
                        userSelect: 'none',
                        color: item.color || 'inherit',
                        backgroundColor: item.background || 'transparent',
                        whiteSpace: 'nowrap',
                    }}
                >
                    {item.icon && (
                        <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                            {item.icon}
                        </span>
                    )}
                    <span>{item.label}</span>
                </div>
            );
        }

        // Regular interactive item
        return (
            <PopupMenuItem
                key={index}
                item={item}
                hoverBackground={hoverBackground}
                onClose={onClose}
            />
        );
    };

    return ReactDOM.createPortal(
        <div
            ref={menuRef}
            style={menuStyle}
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.preventDefault()}
        >
            {items.map(renderItem)}
        </div>,
        document.body
    );
};

// ── Individual menu item ──────────────────────────────────────────────────────

const PopupMenuItem = ({ item, hoverBackground, onClose }) => {
    const [hovered, setHovered] = useState(false);
    const isDisabled = !!item.disabled;

    const handleClick = (e) => {
        if (isDisabled) return;
        if (item.allowOpenInNewTab && item.href && (e.ctrlKey || e.metaKey)) {
            window.open(item.href, '_blank');
            onClose();
            return;
        }
        if (item.onClick) item.onClick();
        onClose();
    };

    const computedBg = isDisabled
        ? (item.background || 'transparent')
        : (hovered ? hoverBackground : (item.background || 'transparent'));

    const style = {
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: item.indent ? '0.45rem 0.75rem 0.45rem 1.75rem' : '0.45rem 0.75rem',
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        userSelect: 'none',
        color: isDisabled ? 'var(--blue_primary_disabled)' : (item.color || 'inherit'),
        backgroundColor: computedBg,
        opacity: isDisabled ? 0.65 : 1,
        whiteSpace: 'nowrap',
        transition: 'background-color 0.1s',
    };

    return (
        <div
            style={style}
            onClick={handleClick}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
        >
            {item.icon && (
                <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                    {item.icon}
                </span>
            )}
            <span style={{ flex: 1 }}>{item.label}</span>
            {item.shortcut && (
                <span style={{ fontSize: '0.8em', opacity: 0.55, marginLeft: '1rem', flexShrink: 0 }}>
                    {item.shortcut}
                </span>
            )}
        </div>
    );
};

export default PopupMenu;
