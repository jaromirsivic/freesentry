import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const TextField = ({
    text,
    fontSize = 14,
    color,
    fontColor = '#000000',
    fontFamily = 'Arial, sans-serif',
    fontStyle = 'normal',
    fontWeight = 'normal',
    style = {},
    outline = null
}) => {
    const computedStyle = {
        fontSize: `${fontSize}px`,
        color: color || fontColor,
        fontFamily: fontFamily,
        fontStyle: fontStyle,
        fontWeight: fontWeight,
        pointerEvents: 'none', // Ensure it doesn't interfere with scene interaction
        userSelect: 'none',
        ...getOutlineStyle(outline),
        ...style
    };

    return (
        <div style={computedStyle}>
            {text}
        </div>
    );
};

export default TextField;
