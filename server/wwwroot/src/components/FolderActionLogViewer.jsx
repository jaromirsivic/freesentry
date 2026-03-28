import React, { useState, useEffect, useRef } from 'react';
import ModalWindow from './ModalWindow';
import { apiFetch } from '../lib/api';

const isTaskFinished = (content) => {
    const lines = content.split('\n').filter(l => l.trim() !== '');
    return lines.slice(-2).some(l => l.includes('[TASK FINISHED]'));
};

const logStatusInputStyleBase = {
    marginRight: 'auto',
    border: '1px solid #e2e8f0',
    borderRadius: '0.375rem',
    padding: '0 0.75rem',
    height: '40px',
    fontSize: '0.875rem',
    outline: 'none',
    width: '120px',
};

const logStatusInputStyleLoading = {
    ...logStatusInputStyleBase,
    background: 'var(--red_primary)',
    color: '#ffffff',
    borderColor: 'var(--red_secondary_disabled)',
};

const logStatusInputStyleFinished = {
    ...logStatusInputStyleBase,
    color: 'var(--blue_secondary)',
    background: '#f8fafc',
};

const logStyle = {
    margin: 0,
    fontFamily: 'monospace',
    fontSize: '0.8rem',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    minHeight: '12rem',
    maxHeight: '60vh',
    overflowY: 'auto',
    background: '#0f172a',
    color: '#e2e8f0',
    padding: '0.75rem',
    borderRadius: '0.25rem',
};

/**
 * Reusable log-viewer modal for Folder Actions.
 *
 * While `isOpen` is true and `logPath` is non-empty, the component polls
 * `pollEndpoint` every 2 s, shows a status indicator that alternates between
 * "Working ..." and "Please Wait." on each response, and stops polling
 * automatically when "[TASK FINISHED]" appears in the last two log lines.
 *
 * Props:
 *   isOpen        {boolean}  – controls modal visibility
 *   title         {string}   – modal title
 *   pollEndpoint  {string}   – API endpoint to POST { log_path } to
 *   logPath       {string}   – log file path returned by the action API call
 *   error         {string}   – error message to show instead of log (optional)
 *   initialContent{string}   – placeholder shown before first poll (optional)
 *   onClose       {function} – called when user clicks Close
 */
const FolderActionLogViewer = ({
    isOpen,
    title,
    pollEndpoint,
    logPath,
    error = '',
    initialContent = 'Starting…',
    onClose,
}) => {
    const [content,    setContent]    = useState('');
    const [statusText, setStatusText] = useState('Working ...');
    const intervalRef  = useRef(null);
    const logEndRef    = useRef(null);

    const stopPolling = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    };

    // Reset content and status each time the modal opens
    useEffect(() => {
        if (isOpen) {
            setContent(initialContent);
            setStatusText('Working ...');
        }
    }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

    // Start / stop polling based on isOpen + logPath
    useEffect(() => {
        if (!isOpen || !logPath) {
            stopPolling();
            return stopPolling;
        }

        const poll = async () => {
            try {
                const data = await apiFetch(pollEndpoint, { log_path: logPath });
                const c = data.content ?? '';
                setContent(c);
                setStatusText(prev => prev === 'Working ...' ? 'Please Wait.' : 'Working ...');
                if (isTaskFinished(c)) {
                    setStatusText('Task Finished');
                    stopPolling();
                }
            } catch {
                // silently ignore poll errors
            }
        };

        poll();
        intervalRef.current = setInterval(poll, 2000);
        return stopPolling;
    }, [isOpen, logPath, pollEndpoint]);

    // Auto-scroll to the latest log line
    useEffect(() => {
        if (logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [content]);

    return (
        <ModalWindow
            isOpen={isOpen}
            title={title}
            okLabel="Close"
            onOk={onClose}
            resizable={true}
            customFooterButtons={[
                <input
                    key="status"
                    type="text"
                    readOnly
                    value={statusText}
                    style={statusText === 'Task Finished' ? logStatusInputStyleFinished : logStatusInputStyleLoading}
                />,
            ]}
        >
            {error
                ? <p style={{ color: 'var(--red_secondary)', margin: 0 }}>{error}</p>
                : (
                    <pre style={logStyle}>
                        {content}
                        <span ref={logEndRef} />
                    </pre>
                )
            }
        </ModalWindow>
    );
};

export default FolderActionLogViewer;
