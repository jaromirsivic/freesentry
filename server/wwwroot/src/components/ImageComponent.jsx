import React from 'react';

const ImageComponent = ({
    src,
    alt = '',
    disabled = false,
    style = {}
}) => {
    const containerStyle = {
        width: '100%',
        opacity: disabled ? 0.5 : 1,
        ...style
    };

    const imgStyle = {
        width: '100%',
        height: 'auto',
        borderRadius: '0.5rem',
        display: 'block'
    };

    return (
        <div className="custom-image" style={containerStyle}>
            <img src={src} alt={alt} style={imgStyle} />
        </div>
    );
};

export default ImageComponent;
