import React, { useRef, useEffect, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import ReactDOM from 'react-dom';
import MarkdownViewer from './MarkdownViewer';
import PopupMenu from './PopupMenu';
import ModalWindow from './ModalWindow';
import { EditorView, lineNumbers, drawSelection, highlightActiveLine, highlightActiveLineGutter, keymap, Decoration } from '@codemirror/view';
import { EditorState, Compartment, StateEffect, StateField, RangeSetBuilder } from '@codemirror/state';
import { syntaxHighlighting, defaultHighlightStyle, indentUnit } from '@codemirror/language';
import { indentMore, indentLess, history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { json } from '@codemirror/lang-json';
import { xml } from '@codemirror/lang-xml';
import { python } from '@codemirror/lang-python';
import { markdown } from '@codemirror/lang-markdown';
import micOffIcon from '../assets/icons/microphone_off.svg';
import micOnIcon from '../assets/icons/microphone_on.svg';
import '../assets/codeEditor/codeEditor.css';

const languageCompartment = new Compartment();
const lineNumbersCompartment = new Compartment();
const lineWrappingCompartment = new Compartment();
const editableCompartment = new Compartment();

const setSearchDecos = StateEffect.define();
const searchDecosField = StateField.define({
    create: () => Decoration.none,
    update(deco, tr) {
        for (const e of tr.effects) {
            if (e.is(setSearchDecos)) return e.value;
        }
        return deco.map(tr.changes);
    },
    provide: f => EditorView.decorations.from(f)
});

const searchMatchMark = Decoration.mark({ class: 'cm-search-match' });
const searchMatchSelectedMark = Decoration.mark({ class: 'cm-search-match cm-search-match-selected' });
const LazyJSXViewer = React.lazy(() => import('./JSXViewer'));

const getLanguageExtension = (language) => {
    switch (language) {
        case 'json':
            return json();
        case 'xml':
            return xml();
        case 'python':
            return python();
        case 'markdown':
            return markdown();
        case 'plaintext':
        default:
            return [];
    }
};

const formatXmlText = (text) => {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('Text is empty');

    const parser = new DOMParser();
    const doc = parser.parseFromString(trimmed, 'application/xml');
    const errorNode = doc.querySelector('parsererror');
    if (errorNode) {
        const errorText = errorNode.textContent || 'Invalid XML';
        const firstLine = errorText.split('\n').find(l => l.trim()) || 'Parse error';
        throw new Error(firstLine.substring(0, 120));
    }

    const serializer = new XMLSerializer();
    let xmlStr = serializer.serializeToString(doc);
    xmlStr = xmlStr.replace(/>\s*</g, '>\n<');

    let indent = 0;
    const formatted = [];

    for (const rawLine of xmlStr.split('\n')) {
        const line = rawLine.trim();
        if (!line) continue;

        if (line.startsWith('</')) {
            indent = Math.max(0, indent - 1);
        }

        formatted.push('    '.repeat(indent) + line);

        if (
            line.startsWith('<') &&
            !line.startsWith('</') &&
            !line.startsWith('<?') &&
            !line.startsWith('<!') &&
            !line.endsWith('/>') &&
            !/<\/[^>]+>$/.test(line)
        ) {
            indent++;
        }
    }

    return formatted.join('\n');
};

const formatJsonText = (text) => {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('Text is empty');
    const parsed = JSON.parse(trimmed);
    return JSON.stringify(parsed, null, 4);
};

const findAllMatches = (docText, query) => {
    if (!query) return [];
    const result = [];
    const lowerDoc = docText.toLowerCase();
    const lowerQuery = query.toLowerCase();
    let idx = 0;
    while (idx <= lowerDoc.length - lowerQuery.length) {
        const pos = lowerDoc.indexOf(lowerQuery, idx);
        if (pos === -1) break;
        result.push({ from: pos, to: pos + query.length });
        idx = pos + query.length;
    }
    return result;
};

const isWhitespaceChar = (char) => char === ' ' || char === '\t' || char === '\n' || char === '\r';

const buildSpeechInsertionText = (doc, from, to, rawTranscript) => {
    const transcript = rawTranscript.trim();
    if (!transcript) return '';

    const prevChar = from > 0 ? doc.sliceString(from - 1, from) : '';
    const nextChar = to < doc.length ? doc.sliceString(to, to + 1) : '';
    const prefix = from > 0 && !isWhitespaceChar(prevChar) ? ' ' : '';
    const suffix = to < doc.length && !isWhitespaceChar(nextChar) ? ' ' : '';

    return prefix + transcript + suffix;
};

const applySearchDecorations = (view, matchList, activeIdx) => {
    if (!view) return;
    if (matchList.length === 0) {
        view.dispatch({ effects: setSearchDecos.of(Decoration.none) });
        return;
    }
    const builder = new RangeSetBuilder();
    for (let i = 0; i < matchList.length; i++) {
        const { from, to } = matchList[i];
        const mark = i === activeIdx ? searchMatchSelectedMark : searchMatchMark;
        builder.add(from, to, mark);
    }
    view.dispatch({ effects: setSearchDecos.of(builder.finish()) });
};

const CodeEditor = forwardRef(({
    enabled = true,
    readonly = false,
    language = 'plaintext',
    showLineNumbers = true,
    showStatusBar = true,
    wordwrap = false,
    value = '',
    onChange,
    title = '',
    style = {},
    rootFolder = '',
    docPath = '',
    showSpeechToTextButton = false,
}, ref) => {
    const editorRef = useRef(null);
    const viewRef = useRef(null);
    const formatErrorTimerRef = useRef(null);
    const searchTextRef = useRef('');
    const matchesRef = useRef([]);
    const currentMatchIdxRef = useRef(-1);
    const searchInputRef = useRef(null);
    const replaceInputRef = useRef(null);

    const [isFocused, setIsFocused] = useState(false);
    const [cursorInfo, setCursorInfo] = useState({ lines: 1, row: 1, column: 1 });
    const [hasSelection, setHasSelection] = useState(false);
    const [contextMenu, setContextMenu] = useState({ visible: false, x: 0, y: 0 });
    const [formatError, setFormatError] = useState(null);
    const [markdownOpen, setMarkdownOpen] = useState(false);
    const [markdownPreviewContent, setMarkdownPreviewContent] = useState('');
    const [jsxOpen, setJsxOpen] = useState(false);
    const [jsxPreviewContent, setJsxPreviewContent] = useState('');

    const [searchPanel, setSearchPanel] = useState(null); // null | 'find' | 'replace'
    const [searchText, setSearchText] = useState('');
    const [replaceText, setReplaceText] = useState('');
    const [matches, setMatches] = useState([]);
    const [, setCurrentMatchIdx] = useState(-1);

    const [showMicButton, setShowMicButton] = useState(showSpeechToTextButton);
    const [listening, setListening] = useState(false);
    const [interim, setInterim] = useState('');
    const [speechErrorModal, setSpeechErrorModal] = useState(false);
    const recognitionRef = useRef(null);
    const lastResultIdxRef = useRef(0);

    useImperativeHandle(ref, () => ({
        focusAndSelectAll: () => {
            const view = viewRef.current;
            if (!view) return;
            view.dispatch({
                selection: { anchor: 0, head: view.state.doc.length }
            });
            view.focus();
        },
        insertText: (text) => {
            const view = viewRef.current;
            if (!view) return;
            const sel = view.state.selection.main;
            view.dispatch({
                changes: { from: sel.from, to: sel.to, insert: text },
                selection: { anchor: sel.from + text.length }
            });
            view.focus();
        }
    }), []);

    const updateCursorInfo = useCallback((state) => {
        const doc = state.doc;
        const selection = state.selection.main;
        const line = doc.lineAt(selection.head);
        setCursorInfo({
            lines: doc.lines,
            row: line.number,
            column: selection.head - line.from + 1
        });
        setHasSelection(selection.from !== selection.to);
    }, []);

    const getEditorText = useCallback(() => {
        return viewRef.current?.state.doc.toString() ?? '';
    }, []);

    const closeContextMenu = useCallback(() => {
        setContextMenu({ visible: false, x: 0, y: 0 });
    }, []);

    const showFormatError = useCallback((message) => {
        if (formatErrorTimerRef.current) clearTimeout(formatErrorTimerRef.current);
        setFormatError(message);
        formatErrorTimerRef.current = setTimeout(() => setFormatError(null), 5000);
    }, []);

    const closeSearchPanel = useCallback(() => {
        setSearchPanel(null);
        setSearchText('');
        setReplaceText('');
        setMatches([]);
        setCurrentMatchIdx(-1);
        searchTextRef.current = '';
        matchesRef.current = [];
        currentMatchIdxRef.current = -1;
        const view = viewRef.current;
        if (view) {
            view.dispatch({ effects: setSearchDecos.of(Decoration.none) });
            view.focus();
        }
    }, []);

    const openFindPanel = useCallback(() => {
        setSearchPanel('find');
        setReplaceText('');
        setTimeout(() => searchInputRef.current?.focus(), 0);
    }, []);

    const openReplacePanel = useCallback(() => {
        setSearchPanel('replace');
        setTimeout(() => searchInputRef.current?.focus(), 0);
    }, []);

    const navigateToMatch = useCallback((idx, matchList) => {
        const view = viewRef.current;
        if (!view || !matchList || matchList.length === 0) return;
        const clampedIdx = ((idx % matchList.length) + matchList.length) % matchList.length;
        setCurrentMatchIdx(clampedIdx);
        currentMatchIdxRef.current = clampedIdx;
        applySearchDecorations(view, matchList, clampedIdx);
        const { from, to } = matchList[clampedIdx];
        view.dispatch({
            selection: { anchor: from, head: to },
            scrollIntoView: true
        });
    }, []);

    const handleSearchTextChange = useCallback((text) => {
        setSearchText(text);
        searchTextRef.current = text;
        const view = viewRef.current;
        if (!view) return;
        const docText = view.state.doc.toString();
        const newMatches = findAllMatches(docText, text);
        matchesRef.current = newMatches;
        setMatches(newMatches);
        if (newMatches.length > 0) {
            const idx = 0;
            setCurrentMatchIdx(idx);
            currentMatchIdxRef.current = idx;
            applySearchDecorations(view, newMatches, idx);
            const { from, to } = newMatches[idx];
            view.dispatch({ selection: { anchor: from, head: to }, scrollIntoView: true });
        } else {
            setCurrentMatchIdx(-1);
            currentMatchIdxRef.current = -1;
            applySearchDecorations(view, [], -1);
        }
    }, []);

    const handleFindNext = useCallback(() => {
        const curMatches = matchesRef.current;
        const curIdx = currentMatchIdxRef.current;
        if (curMatches.length === 0) return;
        navigateToMatch(curIdx + 1, curMatches);
    }, [navigateToMatch]);

    const handleFindPrev = useCallback(() => {
        const curMatches = matchesRef.current;
        const curIdx = currentMatchIdxRef.current;
        if (curMatches.length === 0) return;
        navigateToMatch(curIdx - 1, curMatches);
    }, [navigateToMatch]);

    const handleReplaceOne = useCallback(() => {
        const view = viewRef.current;
        if (!view) return;
        const curMatches = matchesRef.current;
        const curIdx = currentMatchIdxRef.current;
        if (curMatches.length === 0 || curIdx < 0) return;
        const { from, to } = curMatches[curIdx];
        view.dispatch({
            changes: { from, to, insert: replaceText },
        });
        const newDocText = view.state.doc.toString();
        const newMatches = findAllMatches(newDocText, searchTextRef.current);
        matchesRef.current = newMatches;
        setMatches(newMatches);
        if (newMatches.length > 0) {
            const nextIdx = curIdx < newMatches.length ? curIdx : 0;
            setCurrentMatchIdx(nextIdx);
            currentMatchIdxRef.current = nextIdx;
            applySearchDecorations(view, newMatches, nextIdx);
            const m = newMatches[nextIdx];
            view.dispatch({ selection: { anchor: m.from, head: m.to }, scrollIntoView: true });
        } else {
            setCurrentMatchIdx(-1);
            currentMatchIdxRef.current = -1;
            applySearchDecorations(view, [], -1);
        }
    }, [replaceText]);

    const handleReplaceAll = useCallback(() => {
        const view = viewRef.current;
        if (!view) return;
        const curMatches = matchesRef.current;
        if (curMatches.length === 0) return;
        const changes = curMatches.map(({ from, to }) => ({ from, to, insert: replaceText }));
        view.dispatch({ changes });
        const newDocText = view.state.doc.toString();
        const newMatches = findAllMatches(newDocText, searchTextRef.current);
        matchesRef.current = newMatches;
        setMatches(newMatches);
        setCurrentMatchIdx(-1);
        currentMatchIdxRef.current = -1;
        applySearchDecorations(view, newMatches, -1);
    }, [replaceText]);

    const handleCut = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        const sel = view.state.selection.main;
        if (sel.from === sel.to) return;
        const selectedText = view.state.sliceDoc(sel.from, sel.to);
        navigator.clipboard.writeText(selectedText).catch(() => {});
        view.dispatch({
            changes: { from: sel.from, to: sel.to, insert: '' },
            selection: { anchor: sel.from }
        });
        view.focus();
    }, [closeContextMenu]);

    const handleCopy = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        const sel = view.state.selection.main;
        if (sel.from === sel.to) return;
        const selectedText = view.state.sliceDoc(sel.from, sel.to);
        navigator.clipboard.writeText(selectedText).catch(() => {});
        view.focus();
    }, [closeContextMenu]);

    const handlePaste = useCallback(async () => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        try {
            const text = await navigator.clipboard.readText();
            const sel = view.state.selection.main;
            view.dispatch({
                changes: { from: sel.from, to: sel.to, insert: text },
                selection: { anchor: sel.from + text.length }
            });
        } catch { /* clipboard access denied */ }
        view.focus();
    }, [closeContextMenu]);

    const handleSelectAll = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
            selection: { anchor: 0, head: view.state.doc.length }
        });
        view.focus();
    }, [closeContextMenu]);

    const handleDelete = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        const sel = view.state.selection.main;
        if (sel.from === sel.to) return;
        view.dispatch({
            changes: { from: sel.from, to: sel.to, insert: '' },
            selection: { anchor: sel.from }
        });
        view.focus();
    }, [closeContextMenu]);

    const handleFormatXml = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        const text = view.state.doc.toString();
        try {
            const formatted = formatXmlText(text);
            view.dispatch({
                changes: { from: 0, to: text.length, insert: formatted }
            });
        } catch (e) {
            showFormatError('XML formatting failed: ' + (e.message || 'Unknown error'));
        }
        view.focus();
    }, [closeContextMenu, showFormatError]);

    const handleFormatJson = useCallback(() => {
        closeContextMenu();
        const view = viewRef.current;
        if (!view) return;
        const text = view.state.doc.toString();
        try {
            const formatted = formatJsonText(text);
            view.dispatch({
                changes: { from: 0, to: text.length, insert: formatted }
            });
        } catch (e) {
            showFormatError('JSON formatting failed: ' + (e.message || 'Unknown error'));
        }
        view.focus();
    }, [closeContextMenu, showFormatError]);

    const handleViewMarkdown = useCallback(() => {
        closeContextMenu();
        setMarkdownPreviewContent(getEditorText());
        setMarkdownOpen(true);
    }, [closeContextMenu, getEditorText]);

    const handleViewJsx = useCallback(() => {
        closeContextMenu();
        setJsxPreviewContent(getEditorText());
        setJsxOpen(true);
    }, [closeContextMenu, getEditorText]);

    const handleFindFromMenu = useCallback(() => {
        closeContextMenu();
        openFindPanel();
    }, [closeContextMenu, openFindPanel]);

    const handleReplaceFromMenu = useCallback(() => {
        closeContextMenu();
        openReplacePanel();
    }, [closeContextMenu, openReplacePanel]);

    const handleToggleSpeechToText = useCallback(() => {
        closeContextMenu();
        setShowMicButton(prev => {
            if (prev) {
                recognitionRef.current?.stop();
                return false;
            }
            return true;
        });
    }, [closeContextMenu]);

    const handleMicClick = useCallback(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setSpeechErrorModal(true);
            return;
        }
        if (listening) {
            recognitionRef.current?.stop();
            return;
        }
        setInterim('');
        lastResultIdxRef.current = 0;
        const rec = new SpeechRecognition();
        rec.lang = navigator.language || 'cs-CZ';
        rec.continuous = true;
        rec.interimResults = true;
        recognitionRef.current = rec;
        rec.onstart = () => setListening(true);
        rec.onend = () => { setListening(false); setInterim(''); };
        rec.onerror = () => { setListening(false); setInterim(''); };
        rec.onresult = (event) => {
            let interimText = '';
            for (let i = lastResultIdxRef.current; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    const view = viewRef.current;
                    if (view) {
                        const sel = view.state.selection.main;
                        const doc = view.state.doc;
                        const text = buildSpeechInsertionText(doc, sel.from, sel.to, result[0].transcript);
                        if (text) {
                            view.dispatch({
                                changes: { from: sel.from, to: sel.to, insert: text },
                                selection: { anchor: sel.from + text.length }
                            });
                        }
                    }
                    lastResultIdxRef.current = i + 1;
                } else {
                    interimText += result[0].transcript;
                }
            }
            setInterim(interimText);
        };
        rec.start();
    }, [listening]);

    useEffect(() => {
        return () => { recognitionRef.current?.stop(); };
    }, []);

    useEffect(() => {
        if (!editorRef.current) return;

        const updateListener = EditorView.updateListener.of((update) => {
            updateCursorInfo(update.state);
            if (update.docChanged && onChange) {
                const newValue = update.state.doc.toString();
                onChange(newValue);
            }
            if (update.docChanged && searchTextRef.current) {
                const newDocText = update.state.doc.toString();
                const newMatches = findAllMatches(newDocText, searchTextRef.current);
                matchesRef.current = newMatches;
                setMatches(newMatches);
                const prevIdx = currentMatchIdxRef.current;
                const newIdx = newMatches.length > 0 ? Math.min(prevIdx < 0 ? 0 : prevIdx, newMatches.length - 1) : -1;
                currentMatchIdxRef.current = newIdx;
                setCurrentMatchIdx(newIdx);
                applySearchDecorations(update.view, newMatches, newIdx);
            }
        });

        const domHandlers = EditorView.domEventHandlers({
            focus: () => {
                setIsFocused(true);
                return false;
            },
            blur: () => {
                setIsFocused(false);
                return false;
            },
            contextmenu: (e) => {
                e.preventDefault();
                const menuWidth = 210;
                const menuHeight = 380;
                let x = e.clientX;
                let y = e.clientY;
                if (x + menuWidth > window.innerWidth) x = window.innerWidth - menuWidth;
                if (y + menuHeight > window.innerHeight) y = window.innerHeight - menuHeight;
                x = Math.max(0, x);
                y = Math.max(0, y);
                setContextMenu({ visible: true, x, y });
                return true;
            }
        });

        const extensions = [
            languageCompartment.of(getLanguageExtension(language)),
            lineNumbersCompartment.of(showLineNumbers ? lineNumbers() : []),
            lineWrappingCompartment.of(wordwrap ? EditorView.lineWrapping : []),
            editableCompartment.of(EditorView.editable.of(!readonly)),
            syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
            indentUnit.of('    '),
            history(),
            searchDecosField,
            keymap.of([
                { key: 'Ctrl-f', run: () => { openFindPanel(); return true; } },
                { key: 'Ctrl-h', run: () => { openReplacePanel(); return true; } },
                { key: 'Escape', run: () => {
                    if (searchTextRef.current !== '' || document.querySelector('.code-editor-find-panel')) {
                        closeSearchPanel();
                        return true;
                    }
                    return false;
                }},
                ...defaultKeymap,
                ...historyKeymap,
                { key: 'Tab', run: indentMore },
                { key: 'Shift-Tab', run: indentLess }
            ]),
            drawSelection(),
            highlightActiveLine(),
            highlightActiveLineGutter(),
            updateListener,
            domHandlers,
            EditorView.theme({
                '&': {
                    height: '100%'
                },
                '.cm-scroller': {
                    overflow: 'auto'
                },
                '.cm-selectionBackground': {
                    background: '#bfdbfe',
                },
                '&.cm-focused .cm-selectionBackground': {
                    background: '#93c5fd',
                },
                '&.cm-focused > .cm-scroller > .cm-content > .cm-line > .cm-selectionBackground': {
                    background: '#93c5fd',
                },
            })
        ];

        const state = EditorState.create({
            doc: value,
            extensions
        });

        const view = new EditorView({
            state,
            parent: editorRef.current
        });

        viewRef.current = view;
        updateCursorInfo(state);

        return () => {
            view.destroy();
            viewRef.current = null;
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;

        const currentValue = view.state.doc.toString();
        if (value !== currentValue) {
            view.dispatch({
                changes: {
                    from: 0,
                    to: currentValue.length,
                    insert: value
                }
            });
        }
    }, [value]);

    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
            effects: languageCompartment.reconfigure(getLanguageExtension(language))
        });
    }, [language]);

    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
            effects: lineNumbersCompartment.reconfigure(showLineNumbers ? lineNumbers() : [])
        });
    }, [showLineNumbers]);

    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
            effects: lineWrappingCompartment.reconfigure(wordwrap ? EditorView.lineWrapping : [])
        });
    }, [wordwrap]);

    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
            effects: editableCompartment.reconfigure(EditorView.editable.of(!readonly))
        });
    }, [readonly]);

    useEffect(() => {
        return () => {
            if (formatErrorTimerRef.current) clearTimeout(formatErrorTimerRef.current);
        };
    }, []);

    const containerClasses = [
        'code-editor-container',
        isFocused ? 'focused' : '',
        !enabled ? 'disabled' : '',
        readonly ? 'readonly' : ''
    ].filter(Boolean).join(' ');

    const containerStyle = {
        height: '300px',
        ...style,
        position: 'relative'
    };

    const isEditable = !readonly;

    const matchCountLabel = searchText
        ? matches.length === 0
            ? 'No results'
            : `${matches.length} result${matches.length === 1 ? '' : 's'}`
        : '';

    return (
        <div className={containerClasses} style={containerStyle}>
            <div
                ref={editorRef}
                className="code-editor-content"
                style={{ flex: 1, overflow: 'hidden' }}
            />
            {showStatusBar && (
                <div className="code-editor-status-bar">
                    <span className="code-editor-status-interim">{interim}</span>
                    <div className="code-editor-status-stats">
                        <span>Lines: {cursorInfo.lines}</span>
                        <span>Row: {cursorInfo.row}</span>
                        <span>Column: {cursorInfo.column}</span>
                    </div>
                </div>
            )}
            {formatError && (
                <div className="code-editor-format-error">
                    {formatError}
                </div>
            )}
            {searchPanel && (
                <div className="code-editor-find-panel">
                    <div className="code-editor-find-row">
                        <input
                            ref={searchInputRef}
                            className="code-editor-find-input"
                            type="text"
                            placeholder="Find"
                            value={searchText}
                            onChange={e => handleSearchTextChange(e.target.value)}
                            onKeyDown={e => {
                                if (e.key === 'Escape') { closeSearchPanel(); }
                                else if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); handleFindPrev(); }
                                else if (e.key === 'Enter') { e.preventDefault(); handleFindNext(); }
                            }}
                        />
                        {matchCountLabel && (
                            <span className={`code-editor-find-count${matches.length === 0 ? ' no-results' : ''}`}>
                                {matchCountLabel}
                            </span>
                        )}
                        <button className="code-editor-find-btn" title="Previous (Shift+Enter)" onClick={handleFindPrev} disabled={matches.length === 0}>↑</button>
                        <button className="code-editor-find-btn" title="Next (Enter)" onClick={handleFindNext} disabled={matches.length === 0}>↓</button>
                        <button className="code-editor-find-btn close" title="Close (Escape)" onClick={closeSearchPanel}>✕</button>
                    </div>
                    {searchPanel === 'replace' && (
                        <div className="code-editor-find-row">
                            <input
                                ref={replaceInputRef}
                                className="code-editor-find-input"
                                type="text"
                                placeholder="Replace"
                                value={replaceText}
                                onChange={e => setReplaceText(e.target.value)}
                                onKeyDown={e => {
                                    if (e.key === 'Escape') closeSearchPanel();
                                    else if (e.key === 'Enter') { e.preventDefault(); handleReplaceOne(); }
                                }}
                            />
                            <button className="code-editor-find-btn action" title="Replace" onClick={handleReplaceOne} disabled={matches.length === 0}>Replace</button>
                            <button className="code-editor-find-btn action" title="Replace All" onClick={handleReplaceAll} disabled={matches.length === 0}>All</button>
                        </div>
                    )}
                </div>
            )}
            {contextMenu.visible && (
                <PopupMenu
                    position={{ x: contextMenu.x, y: contextMenu.y }}
                    onClose={closeContextMenu}
                    minWidth={210}
                    items={[
                        { label: 'Cut',        shortcut: 'Ctrl+X', disabled: !isEditable || !hasSelection, onClick: handleCut        },
                        { label: 'Copy',       shortcut: 'Ctrl+C', disabled: !hasSelection,                onClick: handleCopy       },
                        { label: 'Paste',      shortcut: 'Ctrl+V', disabled: !isEditable,                  onClick: handlePaste      },
                        { label: 'Select All', shortcut: 'Ctrl+A',                                         onClick: handleSelectAll  },
                        { label: 'Delete',     shortcut: 'Del',    disabled: !isEditable || !hasSelection, onClick: handleDelete     },
                        { type: 'separator' },
                        { label: 'Find',    shortcut: 'Ctrl+F', onClick: handleFindFromMenu },
                        { label: 'Replace', shortcut: 'Ctrl+H', disabled: !isEditable, onClick: handleReplaceFromMenu },
                        { type: 'separator' },
                        { label: 'Speech to Text', onClick: handleToggleSpeechToText },
                        { type: 'separator' },
                        { label: 'Format XML',      disabled: !isEditable, onClick: handleFormatXml  },
                        { label: 'Format JSON',     disabled: !isEditable, onClick: handleFormatJson },
                        { type: 'separator' },
                        { label: 'View JSX',                               onClick: handleViewJsx      },
                        { label: 'View Markdown',                          onClick: handleViewMarkdown },
                    ]}
                />
            )}
            {showMicButton && (
                <button
                    className={`code-editor-mic-btn${listening ? ' listening' : ''}`}
                    onClick={handleMicClick}
                    title={listening ? 'Stop' : 'Speech to Text'}
                    style={{ bottom: showStatusBar ? 36 : 8 }}
                >
                    <img src={listening ? micOnIcon : micOffIcon} alt="mic" />
                </button>
            )}
            {speechErrorModal && ReactDOM.createPortal(
                <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9000 }}>
                    <ModalWindow
                        isOpen
                        title="Speech to Text"
                        onOk={() => setSpeechErrorModal(false)}
                        cancelLabel={null}
                    >
                        <p style={{ margin: '0 0 8px' }}>Your browser does not support speech to text. It is not based on Chromium-based engine.</p>
                        {/win/i.test(navigator.userAgent) && !/android/i.test(navigator.userAgent) && (
                            <p style={{ margin: 0 }}>On Windows you can use the keyboard shortcut <strong>Win&nbsp;+&nbsp;H</strong> for speech to text.</p>
                        )}
                        {/android/i.test(navigator.userAgent) && (
                            <p style={{ margin: 0 }}>On Android you can use dictation by tapping the microphone button on the keyboard.</p>
                        )}
                        {/ipad|iphone|ipod/i.test(navigator.userAgent) && (
                            <p style={{ margin: 0 }}>On iOS you can use dictation by tapping the microphone button on the keyboard.</p>
                        )}
                    </ModalWindow>
                </div>,
                document.body
            )}
            {markdownOpen && ReactDOM.createPortal(
                <div style={{
                    position: 'fixed', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    zIndex: 9000,
                }}>
                    <ModalWindow
                        isOpen={markdownOpen}
                        title={title || 'Markdown Preview'}
                        onCancel={() => setMarkdownOpen(false)}
                        cancelLabel="Close"
                        resizable
                        movable
                        style={{
                            width: 'calc(100vw - 48px)',
                            height: 'calc(100vh - 48px)',
                            maxWidth: 'calc(100vw - 48px)',
                            maxHeight: 'calc(100vh - 48px)',
                        }}
                    >
                        <MarkdownViewer
                            value={markdownPreviewContent}
                            style={{ height: '100%', overflowY: 'auto' }}
                            rootFolder={rootFolder}
                            docPath={docPath}
                        />
                    </ModalWindow>
                </div>,
                document.body
            )}
            {jsxOpen && ReactDOM.createPortal(
                <div style={{
                    position: 'fixed', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    zIndex: 9000,
                }}>
                    <ModalWindow
                        isOpen={jsxOpen}
                        title={title || 'JSX Preview'}
                        onCancel={() => setJsxOpen(false)}
                        cancelLabel="Close"
                        resizable
                        movable
                        style={{
                            width: 'calc(100vw - 48px)',
                            height: 'calc(100vh - 48px)',
                            maxWidth: 'calc(100vw - 48px)',
                            maxHeight: 'calc(100vh - 48px)',
                        }}
                    >
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            height: '100%',
                            minHeight: 0,
                            minWidth: 0,
                            overflow: 'auto',
                        }}>
                            <React.Suspense fallback={
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flex: 1,
                                    minHeight: 0,
                                    color: 'var(--blue_secondary)',
                                    fontSize: '0.95rem',
                                }}>
                                    Loading JSX viewer...
                                </div>
                            }>
                                <LazyJSXViewer content={jsxPreviewContent} />
                            </React.Suspense>
                        </div>
                    </ModalWindow>
                </div>,
                document.body
            )}
        </div>
    );
});

export default CodeEditor;
