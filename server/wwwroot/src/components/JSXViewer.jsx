import React, { useEffect, useState } from 'react';
import { apiText } from '../lib/api';
import jsxViewerScope from './jsxViewerScope';

const shellStyle = {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    minHeight: 0,
    height: '100%',
};

const statusWrapStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    minHeight: 0,
    padding: '1rem',
};

const statusCardBaseStyle = {
    width: '100%',
    borderRadius: '0.5rem',
    border: '1px solid #e2e8f0',
    backgroundColor: '#ffffff',
    boxShadow: '0 1px 2px rgba(15, 23, 42, 0.06)',
    padding: '0.875rem 1rem',
};

const statusTitleStyle = {
    margin: '0 0 0.5rem',
    fontSize: '0.9rem',
    fontWeight: 700,
    color: '#0f172a',
};

const statusMessageStyle = {
    margin: 0,
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    color: 'var(--blue_secondary)',
};

function hasInlineContent(content) {
    return content !== undefined && content !== null;
}

function formatErrorMessage(error, fallbackMessage) {
    if (error instanceof Error && error.message) {
        return error.message;
    }

    if (typeof error === 'string' && error.trim()) {
        return error;
    }

    return fallbackMessage;
}

function createCompileError(message) {
    const error = new Error(message);
    error.name = 'JSXCompileError';
    return error;
}

function createRuntimeError(message) {
    const error = new Error(message);
    error.name = 'JSXRuntimeError';
    return error;
}

function renderStatusCard(title, message, isError = false) {
    return (
        <div style={statusWrapStyle}>
            <div
                style={{
                    ...statusCardBaseStyle,
                    ...(isError
                        ? {
                            border: '1px solid rgba(239, 68, 68, 0.25)',
                            backgroundColor: 'rgba(254, 242, 242, 0.95)',
                        }
                        : null),
                }}
            >
                <h3
                    style={{
                        ...statusTitleStyle,
                        color: isError ? 'var(--red_primary)' : statusTitleStyle.color,
                    }}
                >
                    {title}
                </h3>
                <pre
                    style={{
                        ...statusMessageStyle,
                        color: isError ? 'var(--red_primary)' : statusMessageStyle.color,
                    }}
                >
                    {message}
                </pre>
            </div>
        </div>
    );
}

function normalizeDefaultExport(defaultExport) {
    if (defaultExport === undefined || defaultExport === null) {
        throw createRuntimeError('JSX source must provide a default export.');
    }

    if (React.isValidElement(defaultExport)) {
        return function JSXViewerExportedElement() {
            return defaultExport;
        };
    }

    if (typeof defaultExport === 'function' || typeof defaultExport === 'object') {
        return defaultExport;
    }

    throw createRuntimeError('JSX source default export is not a renderable React component.');
}

function validateSourceText(sourceText) {
    if (!sourceText.trim()) {
        throw createCompileError('JSX source is empty.');
    }

    if (/^\s*import\s/m.test(sourceText)) {
        throw createCompileError('JSXViewer nepodporuje vlastní importy. Použijte runtime scope.');
    }

    if (!/\bexport\s+default\b/.test(sourceText)) {
        throw createCompileError('JSX source must contain `export default`.');
    }
}

async function compileSourceToComponent(sourceText) {
    validateSourceText(sourceText);

    const babelModule = await import('@babel/standalone');
    const Babel = babelModule.default ?? babelModule;

    let compiledCode = '';

    try {
        compiledCode = Babel.transform(sourceText, {
            filename: 'JSXViewer.runtime.jsx',
            sourceType: 'module',
            presets: [['react', { runtime: 'classic' }]],
            plugins: ['transform-modules-commonjs'],
        }).code ?? '';
    } catch (error) {
        throw createCompileError(formatErrorMessage(error, 'Failed to compile JSX source.'));
    }

    const scopeNames = Object.keys(jsxViewerScope);
    const scopeValues = scopeNames.map(name => jsxViewerScope[name]);
    const exportsObject = {};
    const moduleObject = { exports: exportsObject };

    let defaultExport;

    try {
        defaultExport = new Function(
            'exports',
            'module',
            ...scopeNames,
            `${compiledCode}
return (module.exports && Object.prototype.hasOwnProperty.call(module.exports, 'default'))
    ? module.exports.default
    : ((exports && Object.prototype.hasOwnProperty.call(exports, 'default'))
        ? exports.default
        : module.exports);`
        )(exportsObject, moduleObject, ...scopeValues);
    } catch (error) {
        throw createRuntimeError(formatErrorMessage(error, 'Failed to execute JSX source.'));
    }

    return normalizeDefaultExport(defaultExport);
}

class JSXViewerErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidCatch(error) {
        this.props.onError?.(error);
    }

    componentDidUpdate(prevProps) {
        if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
            this.setState({ error: null });
        }
    }

    render() {
        if (this.state.error) {
            return this.props.fallback;
        }

        return this.props.children;
    }
}

const JSXViewer = ({ content, contentUrl }) => {
    const [fetchedSource, setFetchedSource] = useState({
        key: '',
        text: '',
        error: '',
    });
    const [compiledResult, setCompiledResult] = useState({
        key: '',
        sourceText: null,
        component: null,
        compileError: '',
        runtimeError: '',
    });
    const [boundaryError, setBoundaryError] = useState({
        key: '',
        sourceText: null,
        message: '',
    });

    const inlineSourceText = hasInlineContent(content) ? String(content) : null;
    const sourceKey = inlineSourceText !== null
        ? `inline:${inlineSourceText}`
        : (contentUrl ? `url:${contentUrl}` : '');
    const needsFetch = inlineSourceText === null && !!contentUrl;
    const hasFreshFetch = needsFetch && fetchedSource.key === sourceKey;
    const fetchError = inlineSourceText !== null
        ? ''
        : (!contentUrl
            ? 'JSXViewer requires either `content` or `contentUrl`.'
            : (hasFreshFetch ? fetchedSource.error : ''));
    const sourceText = inlineSourceText !== null
        ? inlineSourceText
        : (hasFreshFetch && !fetchedSource.error ? fetchedSource.text : null);
    const hasFreshCompile = sourceText !== null
        && compiledResult.key === sourceKey
        && compiledResult.sourceText === sourceText;
    const CompiledComponent = hasFreshCompile ? compiledResult.component : null;
    const compileError = hasFreshCompile ? compiledResult.compileError : '';
    const prepareRuntimeError = hasFreshCompile ? compiledResult.runtimeError : '';
    const renderRuntimeError = boundaryError.key === sourceKey && boundaryError.sourceText === sourceText
        ? boundaryError.message
        : '';
    const runtimeError = renderRuntimeError || prepareRuntimeError;
    const isSourceLoading = needsFetch && !hasFreshFetch;
    const isCompiling = sourceText !== null && !hasFreshCompile;
    const renderKey = `${sourceKey}|${sourceText ?? ''}`;

    useEffect(() => {
        let isCancelled = false;
        const controller = new AbortController();

        if (!needsFetch) {
            return () => {
                isCancelled = true;
                controller.abort();
            };
        }

        apiText(contentUrl, { signal: controller.signal })
            .then(text => {
                if (!isCancelled) {
                    setFetchedSource({
                        key: sourceKey,
                        text,
                        error: '',
                    });
                }
            })
            .catch(error => {
                if (!isCancelled && error?.name !== 'AbortError') {
                    setFetchedSource({
                        key: sourceKey,
                        text: '',
                        error: formatErrorMessage(error, 'Failed to load JSX source.'),
                    });
                }
            });

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [contentUrl, needsFetch, sourceKey]);

    useEffect(() => {
        let isCancelled = false;

        if (sourceText === null) {
            return () => {
                isCancelled = true;
            };
        }

        compileSourceToComponent(sourceText)
            .then(component => {
                if (!isCancelled) {
                    setCompiledResult({
                        key: sourceKey,
                        sourceText,
                        component,
                        compileError: '',
                        runtimeError: '',
                    });
                }
            })
            .catch(error => {
                if (isCancelled) {
                    return;
                }

                const message = formatErrorMessage(error, 'Failed to prepare JSX source.');

                setCompiledResult({
                    key: sourceKey,
                    sourceText,
                    component: null,
                    compileError: error?.name === 'JSXRuntimeError' ? '' : message,
                    runtimeError: error?.name === 'JSXRuntimeError' ? message : '',
                });
            });

        return () => {
            isCancelled = true;
        };
    }, [sourceKey, sourceText]);

    if (isSourceLoading || isCompiling) {
        return (
            <div style={shellStyle}>
                {renderStatusCard(
                    isSourceLoading ? 'Loading JSX source' : 'Compiling JSX source',
                    isSourceLoading ? 'Loading source text from the requested URL...' : 'Transforming JSX with Babel and preparing the component...'
                )}
            </div>
        );
    }

    if (fetchError) {
        return <div style={shellStyle}>{renderStatusCard('JSX load error', fetchError, true)}</div>;
    }

    if (compileError) {
        return <div style={shellStyle}>{renderStatusCard('JSX compile error', compileError, true)}</div>;
    }

    if (runtimeError && !CompiledComponent) {
        return <div style={shellStyle}>{renderStatusCard('JSX runtime error', runtimeError, true)}</div>;
    }

    if (!CompiledComponent) {
        return <div style={shellStyle}>{renderStatusCard('No JSX content', 'Nothing to render.')}</div>;
    }

    return (
        <div style={shellStyle}>
            <JSXViewerErrorBoundary
                resetKey={renderKey}
                onError={(error) => {
                    setBoundaryError({
                        key: sourceKey,
                        sourceText,
                        message: formatErrorMessage(error, 'Failed to render JSX component.'),
                    });
                }}
                fallback={renderStatusCard(
                    'JSX runtime error',
                    runtimeError || 'Failed to render JSX component.',
                    true
                )}
            >
                <CompiledComponent />
            </JSXViewerErrorBoundary>
        </div>
    );
};

export default JSXViewer;
