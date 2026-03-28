import React from 'react';

const Placeholder = ({
    value = '',
    bgColor = '#FFFFBB',
    style = {},
}) => {
    const containerStyle = {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: bgColor,
        border: '2px solid #000000',
        padding: '1rem',
        boxSizing: 'border-box',
        ...style,
    };

    const textStyle = {
        fontWeight: 700,
        margin: 0,
        textAlign: 'center',
    };

    return (
        <div style={containerStyle}>
            <span style={textStyle}>{value}</span>
        </div>
    );
};

export default Placeholder;
