import React, { useRef, useState } from 'react';
import ModalWindow from './ModalWindow';
import Button from './Button';
import { useAuth } from '../contexts/AuthContext';

const inputStyle = {
    width: '100%',
    padding: '0.5rem 0.75rem',
    border: '1px solid #cbd5e1',
    borderRadius: '0.375rem',
    fontSize: '1rem',
    outline: 'none',
    boxSizing: 'border-box',
    fontFamily: 'inherit',
};

const labelStyle = {
    display: 'block',
    marginBottom: '0.25rem',
    fontWeight: 500,
    fontSize: '0.875rem',
    color: '#374151',
};

const helpContentStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    color: '#374151',
    lineHeight: 1.6,
};

const helpCodeStyle = {
    backgroundColor: '#f1f5f9',
    borderRadius: '0.25rem',
    padding: '0.1rem 0.35rem',
    fontFamily: 'monospace',
    fontSize: '0.95em',
};

/** Login modal shown on a blank page until the user authenticates. */
const LoginModal = () => {
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [isHelpOpen, setIsHelpOpen] = useState(false);
    const usernameRef = useRef(null);

    const handleOk = async () => {
        if (!username || !password) return;
        setIsLoading(true);
        setError('');
        try {
            await login(username, password);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setUsername('');
        setPassword('');
        setError('');
        setTimeout(() => usernameRef.current?.focus(), 0);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleOk();
    };

    const resetButton = error ? (
        <Button
            label="Reset"
            onClick={handleReset}
            color="var(--blue_secondary)"
            style={{ height: '40px', display: 'flex', alignItems: 'center' }}
        />
    ) : null;

    return (
        <>
            <ModalWindow
                isOpen
                title="Sign In"
                onOk={handleOk}
                okLabel={isLoading ? 'Signing in…' : 'Sign In'}
                okDisabled={isLoading || !username || !password}
                validationErrors={error ? [error] : []}
                customFooterButtons={resetButton}
                movable={false}
                headerAction={
                    <Button
                        label="Help"
                        onClick={() => setIsHelpOpen(true)}
                        color="var(--blue_secondary)"
                        style={{
                            height: '32px',
                            display: 'flex',
                            alignItems: 'center',
                            padding: '0.35rem 0.75rem',
                            fontSize: '0.875rem',
                        }}
                    />
                }
            >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div>
                        <label style={labelStyle} htmlFor="login-username">
                            Username
                        </label>
                        <input
                            id="login-username"
                            ref={usernameRef}
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            onKeyDown={handleKeyDown}
                            style={inputStyle}
                            autoFocus
                            autoComplete="username"
                        />
                    </div>
                    <div>
                        <label style={labelStyle} htmlFor="login-password">
                            Password
                        </label>
                        <input
                            id="login-password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            onKeyDown={handleKeyDown}
                            style={inputStyle}
                            autoComplete="current-password"
                        />
                    </div>
                </div>
            </ModalWindow>

            {isHelpOpen && (
                <ModalWindow
                    isOpen
                    title="Help"
                    onOk={() => setIsHelpOpen(false)}
                    okLabel="Close"
                    movable={false}
                    style={{ maxWidth: '560px' }}
                >
                    <div style={helpContentStyle}>
                        <p style={{ margin: 0 }}>
                            Use the default credentials below only for the initial sign-in:
                        </p>
                        <div>
                            <div>
                                <strong>Administrator:</strong>{' '}
                                username <code style={helpCodeStyle}>admin</code>, password{' '}
                                <code style={helpCodeStyle}>admin</code>
                            </div>
                            <div>
                                <strong>Limited-access user:</strong>{' '}
                                username <code style={helpCodeStyle}>user</code>, password{' '}
                                <code style={helpCodeStyle}>user</code>
                            </div>
                        </div>
                        <p style={{ margin: 0 }}>
                            Both the administrator and limited-access accounts should have their
                            passwords changed immediately after signing in.
                        </p>
                        <p style={{ margin: 0 }}>
                            If you do not need the default limited-access account, sign in as{' '}
                            <code style={helpCodeStyle}>admin</code>, open{' '}
                            <strong>Menu &gt; Settings</strong>, and remove or update that user in{' '}
                            <code style={helpCodeStyle}>settings.json</code>.
                        </p>
                    </div>
                </ModalWindow>
            )}
        </>
    );
};

export default LoginModal;
