import React from 'react';

const ColumnLayout = ({
    children,
    gap = '1rem',
    style = {}
}) => {
    const columnStyle = {
        display: 'flex',
        flexDirection: 'column',
        gap: gap,
        background: 'transparent',
        width: '100%',
        ...style
    };

    return (
        <div className="custom-column-layout" style={columnStyle}>
            {children}
        </div>
    );
};

export default ColumnLayout;
