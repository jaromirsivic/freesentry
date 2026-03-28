import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { sha256 } from '../lib/sha256';

const AuthContext = createContext(null);

const TOKEN_KEY = 'token';

/** Decode the payload of a JWT without verifying the signature. */
function decodeJwtPayload(token) {
    try {
        const base64 = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
        return JSON.parse(atob(base64));
    } catch {
        return {};
    }
}

/** Provides authentication state (token, role, login, logout) to the component tree. */
export function AuthProvider({ children }) {
    const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));

    const login = useCallback(async (username, password) => {
        const passwordHash = sha256(password);

        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password: passwordHash }),
        });
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || 'Login failed.');
        }
        const { token: newToken } = await response.json();
        localStorage.setItem(TOKEN_KEY, newToken);
        setToken(newToken);
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
    }, []);

    useEffect(() => {
        window.addEventListener('token:auth-expired', logout);
        return () => window.removeEventListener('token:auth-expired', logout);
    }, [logout]);

    const role = token ? (decodeJwtPayload(token).role ?? '') : '';

    return (
        <AuthContext.Provider
            value={{ token, role, login, logout, isAuthenticated: !!token }}
        >
            {children}
        </AuthContext.Provider>
    );
}

/** Returns auth context. Must be used inside AuthProvider. */
export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider.');
    return ctx;
}
