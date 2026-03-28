import React from 'react';
import { getOutlineStyle } from '../lib/outlineStyle';

const StaticText = ({
    text,
    disabled = false,
    style = {},
    outline = null
}) => {
    const outlineStyle = getOutlineStyle(outline);
    const textStyle = {
        opacity: disabled ? 0.5 : 1,
        color: 'inherit',
        ...(outline ? { display: 'inline-block' } : {}),
        ...outlineStyle,
        ...style
    };

    return (
        <span style={textStyle} className="custom-static-text">
            {text}
        </span>
    );
};

export default StaticText;
