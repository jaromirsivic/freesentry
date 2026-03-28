import React from 'react';

const HorizontalSeparator = ({ label, fullWidth = false, bleed = '1.5rem', color = '#3b82f6', help }) => {
    const containerStyle = {
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        ...(fullWidth ? {
            width: `calc(100% + calc(${bleed} * 2))`,
            marginLeft: `-${bleed}`,
            marginRight: `-${bleed}`,
            paddingLeft: '0', // Ensure line touches the left edge
            paddingRight: '0'
        } : {
            width: '100%'
        })
    };

    return (
        <div style={containerStyle}>
            <div style={{ width: fullWidth ? '1.5rem' : '1rem', height: '1px', backgroundColor: '#ddddddff' }}></div>
            {label && (
                <span style={{ fontWeight: '500', color: color, whiteSpace: 'nowrap' }}>
                    {label}
                    {help && (
                        <>
                            &nbsp; &#10914; &nbsp;
                            <a
                                href={help}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{ color: 'inherit', textDecoration: 'underline' }}
                            >
                                more info
                            </a>
                        </>
                    )}
                </span>
            )}
            <div style={{ flexGrow: 1, height: '1px', backgroundColor: '#ddddddff' }}></div>
        </div>
    );
};

export default HorizontalSeparator;
