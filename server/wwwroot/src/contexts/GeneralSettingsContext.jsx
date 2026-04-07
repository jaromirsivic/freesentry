import React, { createContext, useContext, useEffect, useState } from 'react';
import { getGeneralSettings } from '../lib/api';

const GeneralSettingsContext = createContext({
    isDebugMode: false,
    generalSettingsLoaded: false,
});

/** Loads general settings once and exposes whether app UI is in debug mode (`general.mode === 'debug'`). */
export function GeneralSettingsProvider({ children }) {
    const [isDebugMode, setIsDebugMode] = useState(false);
    const [generalSettingsLoaded, setGeneralSettingsLoaded] = useState(false);

    useEffect(() => {
        let cancelled = false;
        getGeneralSettings()
            .then((data) => {
                if (!cancelled) {
                    setIsDebugMode(data?.mode === 'debug');
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setIsDebugMode(false);
                }
            })
            .finally(() => {
                if (!cancelled) {
                    setGeneralSettingsLoaded(true);
                }
            });
        return () => {
            cancelled = true;
        };
    }, []);

    return (
        <GeneralSettingsContext.Provider value={{ isDebugMode, generalSettingsLoaded }}>
            {children}
        </GeneralSettingsContext.Provider>
    );
}

export function useGeneralSettings() {
    return useContext(GeneralSettingsContext);
}
