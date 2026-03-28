/**
 * Returns CSS style properties for rendering a component outline border.
 * @param {object|null} outline - Outline config: { color, width, style, radius }
 * @returns {object} CSS style properties object
 */
export function getOutlineStyle(outline) {
    if (!outline) return {};
    const {
        color = 'var(--blue_primary)',
        width = 2,
        style = 'solid',
        radius,
    } = outline;
    return {
        border: `${width}px ${style} ${color}`,
        ...(radius !== undefined ? { borderRadius: `${radius}px` } : {}),
    };
}
