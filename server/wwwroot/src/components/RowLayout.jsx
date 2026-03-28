import React from 'react';

const RowLayout = ({
    children,
    gap = '1rem',
    style = {}
}) => {
    const rowStyle = {
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: gap,
        background: 'transparent',
        width: '100%',
        ...style
    };

    return (
        <div className="custom-row-layout" style={rowStyle}>
            {children}
        </div>
    );
};

export default RowLayout;
