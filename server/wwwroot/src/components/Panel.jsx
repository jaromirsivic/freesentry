import React from 'react';

const Panel = ({
    children,
    backgroundColor = '#ffffff',
    textColor,
    enabled = true,
    style = {},
    title,
    headerAction
}) => {
    const panelStyle = {
        backgroundColor: backgroundColor,
        color: textColor || 'inherit',
        borderRadius: '0.5rem',
        boxShadow: '4px 4px 10px rgba(0, 0, 0, 0.1)',
        opacity: enabled ? 1 : 0.6,
        pointerEvents: enabled ? 'auto' : 'none',
        transition: 'all 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden', // Ensure content doesn't overflow rounded corners
        ...style
    };

    return (
        <div className="custom-panel" style={panelStyle}>
            {(title || headerAction) && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: '1px solid #e5e7eb',
                    padding: '0.5rem 1.0rem',
                    backgroundColor: '#E9E9E9'
                }}>
                    {title && <h2 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 700 }}>{title}</h2>}
                    {headerAction && <div>{headerAction}</div>}
                </div>
            )}
            <div style={{ padding: '1.5rem' }}>
                {children}
            </div>
        </div>
    );
};

export default Panel;
