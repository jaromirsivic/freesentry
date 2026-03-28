import React, { createContext, useContext, useEffect, useState } from 'react';

const ConfigContext = createContext({ runMode: 'development', configLoaded: false });

/** Fetches public app config from the backend and exposes it to the tree. */
export function ConfigProvider({ children }) {
    const [runMode, setRunMode] = useState('development');
    const [configLoaded, setConfigLoaded] = useState(false);

    useEffect(() => {
        fetch('/api/settings/config')
            .then((r) => r.json())
            .then((data) => setRunMode(data.runMode || 'development'))
            .catch(() => {})
            .finally(() => setConfigLoaded(true));
    }, []);

    return (
        <ConfigContext.Provider value={{ runMode, configLoaded }}>
            {children}
        </ConfigContext.Provider>
    );
}

/** Returns the app config context. */
export function useConfig() {
    return useContext(ConfigContext);
}
