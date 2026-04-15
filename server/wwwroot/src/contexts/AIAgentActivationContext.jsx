/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import {
    activateAIAgent as activateAIAgentApi,
    getAIAgentActivation,
    deactivateAIAgent as deactivateAIAgentApi
} from '../lib/api';

const AIAgentActivationContext = createContext({
    aiagentFullyActivated: false,
    activationLoaded: false,
    isActivationBusy: false,
    refreshActivation: async () => false,
    activateAIAgent: async () => false,
    deactivateAIAgent: async () => false
});

export function AIAgentActivationProvider({ children }) {
    const [aiagentFullyActivated, setAIAgentFullyActivated] = useState(false);
    const [activationLoaded, setActivationLoaded] = useState(false);
    const [isActivationBusy, setIsActivationBusy] = useState(false);

    const refreshActivation = useCallback(async () => {
        const data = await getAIAgentActivation();
        const nextState = data?.aiagent_fully_activated === true;
        setAIAgentFullyActivated(nextState);
        return nextState;
    }, []);

    useEffect(() => {
        let cancelled = false;

        refreshActivation()
            .catch((error) => {
                if (!cancelled) {
                    console.error('Failed to load AI agent activation state:', error);
                    setAIAgentFullyActivated(false);
                }
            })
            .finally(() => {
                if (!cancelled) {
                    setActivationLoaded(true);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [refreshActivation]);

    const activateAIAgent = useCallback(async () => {
        setIsActivationBusy(true);
        try {
            const data = await activateAIAgentApi();
            const nextState = data?.aiagent_fully_activated === true;
            setAIAgentFullyActivated(nextState);
            return nextState;
        } catch (error) {
            console.error('Failed to activate AI agent:', error);
            throw error;
        } finally {
            setIsActivationBusy(false);
        }
    }, []);

    const deactivateAIAgent = useCallback(async () => {
        setIsActivationBusy(true);
        try {
            const data = await deactivateAIAgentApi();
            const nextState = data?.aiagent_fully_activated === true;
            setAIAgentFullyActivated(nextState);
            return nextState;
        } catch (error) {
            console.error('Failed to deactivate AI agent:', error);
            throw error;
        } finally {
            setIsActivationBusy(false);
        }
    }, []);

    const value = useMemo(() => ({
        aiagentFullyActivated,
        activationLoaded,
        isActivationBusy,
        refreshActivation,
        activateAIAgent,
        deactivateAIAgent
    }), [
        aiagentFullyActivated,
        activationLoaded,
        isActivationBusy,
        refreshActivation,
        activateAIAgent,
        deactivateAIAgent
    ]);

    return (
        <AIAgentActivationContext.Provider value={value}>
            {children}
        </AIAgentActivationContext.Provider>
    );
}

export function useAIAgentActivation() {
    return useContext(AIAgentActivationContext);
}
