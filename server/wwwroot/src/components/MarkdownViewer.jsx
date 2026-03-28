import React, { useEffect, useRef } from 'react';
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import markedKatex from 'marked-katex-extension';
import DOMPurify from 'dompurify';
import mermaid from 'mermaid';
import hljs from 'highlight.js';
import { useNavigate } from 'react-router-dom';
import 'highlight.js/styles/github.css';
import 'katex/dist/katex.min.css';
import '../assets/markdownViewer/markdownViewer.css';
import { apiBlob } from '../lib/api';

function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function createInlineWrapperExtension({ name, markerStart, markerEnd, tagName, className = '' }) {
    const rule = new RegExp(`^${escapeRegExp(markerStart)}([\\s\\S]+?)${escapeRegExp(markerEnd)}`);

    return {
        name,
        level: 'inline',
        start(src) {
            const index = src.indexOf(markerStart);
            return index >= 0 ? index : undefined;
        },
        tokenizer(src) {
            const match = rule.exec(src);
            if (!match || !match[1].trim()) return undefined;

            return {
                type: name,
                raw: match[0],
                text: match[1],
                tokens: this.lexer.inlineTokens(match[1]),
            };
        },
        renderer(token) {
            const classAttr = className ? ` class="${className}"` : '';
            return `<${tagName}${classAttr}>${this.parser.parseInline(token.tokens)}</${tagName}>`;
        },
        childTokens: ['tokens'],
    };
}

const markdownParser = new Marked(
    markedHighlight({
        langPrefix: 'hljs language-',
        highlight(code, lang) {
            if (lang === 'mermaid') return code;
            const language = hljs.getLanguage(lang) ? lang : null;
            return language
                ? hljs.highlight(code, { language }).value
                : code;
        },
    }),
    markedKatex({
        throwOnError: false,
        nonStandard: true,
    }),
    {
        extensions: [
            createInlineWrapperExtension({
                name: 'highlight',
                markerStart: '==',
                markerEnd: '==',
                tagName: 'mark',
                className: 'markdown-highlight',
            }),
            createInlineWrapperExtension({
                name: 'critic-addition-braces',
                markerStart: '{+',
                markerEnd: '+}',
                tagName: 'ins',
                className: 'critic-addition',
            }),
            createInlineWrapperExtension({
                name: 'critic-addition-brackets',
                markerStart: '[+',
                markerEnd: '+]',
                tagName: 'ins',
                className: 'critic-addition',
            }),
            createInlineWrapperExtension({
                name: 'critic-deletion-braces',
                markerStart: '{-',
                markerEnd: '-}',
                tagName: 'del',
                className: 'critic-deletion',
            }),
            createInlineWrapperExtension({
                name: 'critic-deletion-brackets',
                markerStart: '[-',
                markerEnd: '-]',
                tagName: 'del',
                className: 'critic-deletion',
            }),
        ],
    },
);

mermaid.initialize({ startOnLoad: false, theme: 'default' });

/** Decode HTML entities produced by `marked` inside fenced code blocks. */
function decodeEntities(s) {
    return s
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
}

/**
 * Replace `marked`-generated <pre><code class="language-mermaid"> blocks
 * with <div class="mermaid"> so the Mermaid library can process them.
 */
function extractMermaidBlocks(html) {
    return html.replace(
        /<pre><code[^>]*class="[^"]*language-mermaid[^"]*"[^>]*>([\s\S]*?)<\/code><\/pre>/g,
        (_, code) => `<div class="mermaid">${decodeEntities(code)}</div>`,
    );
}

/** Return true if a src value is an absolute URL that needs no patching. */
function isAbsoluteSrc(src) {
    return src.startsWith('//') || /^[a-zA-Z][a-zA-Z\d+.-]*:/.test(src);
}

/** Resolve a relative file reference against the current document folder. */
function resolveRelativePath(rawValue, docPath = '') {
    try {
        const normalizedDocPath = docPath.replace(/^\/+|\/+$/g, '');
        const base = new URL(`http://x/${normalizedDocPath ? `${normalizedDocPath}/` : ''}`);
        const resolved = new URL(rawValue, base);
        const parts = resolved.pathname.split('/').filter(Boolean).map(part => decodeURIComponent(part));
        const name = parts.pop();

        if (!name) return null;

        return {
            path: parts.join('/'),
            name,
            hash: resolved.hash,
        };
    } catch {
        return null;
    }
}

/** Return true if href points to another markdown file within the same rootFolder. */
function isRelativeMarkdownHref(href) {
    if (!href || href.startsWith('#') || href.startsWith('?') || href.startsWith('/')) return false;
    if (isAbsoluteSrc(href)) return false;

    try {
        return new URL(href, 'http://x/').pathname.toLowerCase().endsWith('.md');
    } catch {
        return false;
    }
}

function buildMarkdownViewUrl({ rootFolder, path, name, hash = '' }) {
    const params = new URLSearchParams({ rootFolder, path, name });
    return `/markdownview?${params.toString()}${hash}`;
}

/**
 * Renders Markdown text as styled HTML, including Mermaid diagrams and KaTeX math.
 *
 * @param {string} value      - Markdown source. Defaults to "".
 * @param {object} style      - Additional inline styles for the container.
 * @param {object} scrollRef  - Optional ref object that will be pointed at the
 *                              container DOM node (e.g. to let the parent scroll it).
 * @param {string} rootFolder - Root folder key used to resolve relative images and links.
 * @param {string} docPath    - Path of the document's parent folder within rootFolder.
 */
const MarkdownViewer = ({ value = '', style = {}, scrollRef = null, rootFolder = '', docPath = '' }) => {
    const navigate = useNavigate();
    const internalRef = useRef(null);
    const containerRef = scrollRef ?? internalRef;
    const blobUrlsRef = useRef([]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const nodes = container.querySelectorAll('.mermaid');
        if (nodes.length === 0) return;

        // Reset previously processed nodes so mermaid re-renders them
        nodes.forEach(node => {
            node.removeAttribute('data-processed');
        });

        mermaid.run({ nodes });
    }, [value, containerRef]);

    useEffect(() => {
        // Revoke any blob URLs created for the previous render
        blobUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
        blobUrlsRef.current = [];

        const container = containerRef.current;
        if (!container || !rootFolder) return;

        const imgs = container.querySelectorAll('img');
        imgs.forEach(img => {
            const rawSrc = img.getAttribute('src');
            if (!rawSrc || isAbsoluteSrc(rawSrc)) return;

            const target = resolveRelativePath(rawSrc, docPath);
            if (!target) return;
            const { path, name } = target;

            apiBlob('/api/folder/download', { rootFolder, path, name })
                .then(blobUrl => {
                    blobUrlsRef.current.push(blobUrl);
                    img.src = blobUrl;
                })
                .catch(() => {
                    // Leave the broken img in place; don't crash the viewer
                });
        });

        return () => {
            blobUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
            blobUrlsRef.current = [];
        };
    }, [value, rootFolder, docPath, containerRef]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const links = container.querySelectorAll('a[href]');
        links.forEach(link => {
            const originalHref = link.dataset.originalMarkdownHref ?? link.getAttribute('href');
            if (link.dataset.originalMarkdownHref) {
                link.setAttribute('href', link.dataset.originalMarkdownHref);
            }
            delete link.dataset.markdownPreviewUrl;

            if (!originalHref || !rootFolder || !isRelativeMarkdownHref(originalHref)) return;

            const target = resolveRelativePath(originalHref, docPath);
            if (!target) return;

            const previewUrl = buildMarkdownViewUrl({ rootFolder, ...target });
            link.dataset.originalMarkdownHref = originalHref;
            link.setAttribute('href', previewUrl);
            link.dataset.markdownPreviewUrl = previewUrl;
        });
    }, [value, rootFolder, docPath, containerRef]);

    const rawHtml = markdownParser.parse(value);
    const htmlWithMermaid = extractMermaidBlocks(rawHtml);

    // Sanitize — allow class attribute so mermaid <div class="mermaid"> is preserved
    const safeHtml = DOMPurify.sanitize(htmlWithMermaid, {
        ADD_ATTR: ['class'],
    });

    const handleViewerClick = (event) => {
        if (event.defaultPrevented || event.button !== 0) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        if (!(event.target instanceof Element)) return;

        const link = event.target.closest('a[data-markdown-preview-url]');
        if (!(link instanceof HTMLAnchorElement)) return;
        if (link.target && link.target !== '_self') return;
        if (link.hasAttribute('download')) return;

        const previewUrl = link.dataset.markdownPreviewUrl;
        if (!previewUrl) return;

        event.preventDefault();
        navigate(previewUrl);
    };

    return (
        <div
            ref={containerRef}
            className="markdown-viewer"
            style={style}
            onClick={handleViewerClick}
            dangerouslySetInnerHTML={{ __html: safeHtml }}
        />
    );

};

export default MarkdownViewer;
