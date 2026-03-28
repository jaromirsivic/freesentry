import React from 'react';
import { Outlet } from 'react-router-dom';

const Tools = () => {
    return (
        <div className="page-container">
            <Outlet />
        </div>
    );
};

export default Tools;
