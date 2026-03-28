/**
 * Converts a hex color to rgba string with the given alpha.
 * Supports #RGB and #RRGGBB. Returns fallback for non-hex colors.
 * @param {string} color - CSS color (e.g. '#f59e0b')
 * @param {number} alpha - Alpha value 0–1
 * @returns {string} rgba(r, g, b, alpha)
 */
export function toRgba(color, alpha = 0.6) {
    if (!color || typeof color !== 'string') return `rgba(0, 0, 0, ${alpha})`;
    const hex = color.trim();
    if (hex[0] !== '#') return `rgba(0, 0, 0, ${alpha})`;
    let r, g, b;
    if (hex.length === 4) {
        r = parseInt(hex[1] + hex[1], 16);
        g = parseInt(hex[2] + hex[2], 16);
        b = parseInt(hex[3] + hex[3], 16);
    } else if (hex.length >= 7) {
        r = parseInt(hex.slice(1, 3), 16);
        g = parseInt(hex.slice(3, 5), 16);
        b = parseInt(hex.slice(5, 7), 16);
    } else {
        return `rgba(0, 0, 0, ${alpha})`;
    }
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return `rgba(0, 0, 0, ${alpha})`;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Returns style object for the outline effect (border + glow).
 * Does not include padding so callers can keep their own padding and avoid size change.
 * @param {string|null|undefined} outline - CSS color or falsy for no outline
 * @returns {object} Style object to spread onto root element
 */
export function getOutlineStyle(outline) {
    if (!outline) return {};
    return {
        border: `2px solid ${outline}`,
        boxShadow: `0 0 8px ${toRgba(outline, 0.6)}`,
        transition: 'all 0.2s ease'
    };
}
