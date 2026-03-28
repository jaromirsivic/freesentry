import React from 'react';
import { Outlet } from 'react-router-dom';

const Sandbox = () => {
    return (
        <div className="page-container">
            <Outlet />
        </div>
    );
};

export default Sandbox;
