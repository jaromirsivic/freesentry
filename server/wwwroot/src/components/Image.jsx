import React from 'react';

const Image = ({
    border = 0,
    src,
    stretchMode = 'fit',
    background = 'transparent',
    style = {}
}) => {
    const getImageStyle = () => {
        switch (stretchMode) {
            case 'fit':
                return {
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                };
            case 'stretch':
                return {
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    objectFit: 'fill'
                };
            case 'originalSize':
                return {
                    display: 'block',
                    width: 'auto',
                    height: 'auto',
                    maxWidth: '100%',
                    maxHeight: '100%'
                };
            default:
                return {
                    display: 'block'
                };
        }
    };

    const containerStyle = {
        border: border > 0 ? `${border}px dashed gray` : 'none',
        display: stretchMode === 'originalSize' ? 'inline-block' : 'block',
        width: stretchMode === 'originalSize' ? 'auto' : '100%',
        height: stretchMode === 'originalSize' ? 'auto' : '100%',
        backgroundColor: background,
        ...style
    };

    return (
        <div className="custom-image" style={containerStyle}>
            <img src={src} alt="" style={getImageStyle()} />
        </div>
    );
};

export default Image;

