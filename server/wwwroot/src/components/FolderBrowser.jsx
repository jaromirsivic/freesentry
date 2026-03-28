import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import ModalWindow from './ModalWindow';
import PopupMenu from './PopupMenu';
import Button from './Button';
import CodeEditor from './CodeEditor';
import ProgressBar from './ProgressBar';
import MultiSwitch from './MultiSwitch';
import Switch from './Switch';
import Slider from './Slider';
import HorizontalSeparator from './HorizontalSeparator';
import Polygon from './Polygon';
import { apiFetch, apiUpload, apiUploadWithProgress, apiDownload, apiBlob } from '../lib/api';
import { isValidFileOrFolderName } from '../lib/fileNameValidation';

// ── Constants ─────────────────────────────────────────────────────────────────

const INITIAL_LIMIT = 1000;

// ── Material Design icon paths (viewBox "0 -960 960 960") ─────────────────────

const ICONS = {
    folder:    'M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v400q0 33-23.5 56.5T800-160H160Zm0-80h640v-400H447l-80-80H160v480Zm0 0v-480 480Z',
    file:      'M320-240h320v-80H320v80Zm0-160h320v-80H320v80ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z',
    upload:    'M440-320v-326L336-542l-56-58 200-200 200 200-56 58-104-104v326h-80ZM240-160v-80h480v80H240Z',
    newFolder: 'M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h240v80H447l-80-80H160v480h640v-240h80v240q0 33-23.5 56.5T800-160H160Zm600-360v-120H640v-80h120v-120h80v120h120v80H840v120h-80ZM160-240v-480 480Z',
    newFile:   'M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h360v80H200v560h560v-360h80v360q0 33-23.5 56.5T760-120H200Zm480-400v-120H560v-80h120v-120h80v120h120v80H760v120h-80ZM200-200v-560 560Z',
    web:       'M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm-40-82v-78q-33 0-56.5-23.5T360-320v-40L168-552q-3 18-5.5 36T160-480q0 136 93 238t187 160Zm284-98q11-15 19.5-38t12.5-42l-67-44q-8 32-16.5 55.5T654-320H600v40q0 17-6.5 30.5T576-228l188 48Z',
    download:  'M480-320 280-520l56-58 104 104v-326h80v326l104-104 56 58-200 200ZM240-160v-80h480v80H240Z',
    rename:    'M200-200h57l391-391-57-57-391 391v57Zm-80 80v-170l528-527q12-11 26.5-17t30.5-6q16 0 31 6t26 18l55 56q12 11 17.5 26t5.5 30q0 16-5.5 30.5T817-647L290-120H120Zm640-584-56-56 56 56Zm-141 85-28-29 57 57-29-28Z',
    move:      'M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z',
    delete:    'M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360ZM280-720v520-520Z',
    search:    'M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z',
    view:      'M480-320q75 0 127.5-52.5T660-500q0-75-52.5-127.5T480-680q-75 0-127.5 52.5T300-500q0 75 52.5 127.5T480-320Zm0-72q-45 0-76.5-31.5T372-500q0-45 31.5-76.5T480-608q45 0 76.5 31.5T588-500q0 45-31.5 76.5T480-392Zm0 192q-146 0-266-81.5T40-500q54-137 174-218.5T480-800q146 0 266 81.5T920-500q-54 137-174 218.5T480-200Zm0-300Zm0 220q113 0 207.5-59.5T832-500q-50-101-144.5-160.5T480-720q-113 0-207.5 59.5T128-500q50 101 144.5 160.5T480-280Z',
    refresh:   'M480-160q-134 0-227-93t-93-227q0-134 93-227t227-93q69 0 132 28.5T720-690v-110h80v280H520v-80h168q-32-56-87.5-88T480-720q-100 0-170 70t-70 170q0 100 70 170t170 70q77 0 139-44t87-116h84q-28 106-114 173t-196 67Z',
    moreVert:  'M480-160q-33 0-56.5-23.5T400-240q0-33 23.5-56.5T480-320q33 0 56.5 23.5T560-240q0 33-23.5 56.5T480-160Zm0-240q-33 0-56.5-23.5T400-480q0-33 23.5-56.5T480-560q33 0 56.5 23.5T560-480q0 33-23.5 56.5T480-400Zm0-240q-33 0-56.5-23.5T400-720q0-33 23.5-56.5T480-800q33 0 56.5 23.5T560-720q0 33-23.5 56.5T480-640Z',
};

/** Renders a Material Design SVG icon inline. */
function Ic({ d, fill = '#374151', size = 16 }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg"
             width={size} height={size}
             viewBox="0 -960 960 960"
             fill={fill}
             style={{ display: 'inline-block', flexShrink: 0, verticalAlign: 'middle' }}>
            <path d={d} />
        </svg>
    );
}

// ── Formatters ────────────────────────────────────────────────────────────────

function formatSize(bytes) {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
    });
}

function sortItems(items, by, dir) {
    const mult = dir === 'asc' ? 1 : -1;
    return [...items].sort((a, b) => {
        // Folders always first
        if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
        let va = a[by] ?? '';
        let vb = b[by] ?? '';
        if (by === 'size') { va = va ?? -1; vb = vb ?? -1; }
        if (typeof va === 'string') return mult * va.localeCompare(vb, undefined, { sensitivity: 'base' });
        return mult * (va < vb ? -1 : va > vb ? 1 : 0);
    });
}

// ── Drag-and-drop directory traversal (webkitGetAsEntry) ─────────────────────

async function readAllEntries(dirReader) {
    let all = [];
    let batch;
    do {
        batch = await new Promise((res, rej) => dirReader.readEntries(res, rej));
        all = all.concat(batch);
    } while (batch.length > 0);
    return all;
}

async function collectFilesFromEntry(entry, basePath = '') {
    if (entry.isFile) {
        const file = await new Promise((res, rej) => entry.file(res, rej));
        return [{ file, relativePath: basePath }];
    }
    if (entry.isDirectory) {
        const subPath = basePath ? `${basePath}/${entry.name}` : entry.name;
        const children = await readAllEntries(entry.createReader());
        const nested = await Promise.all(children.map(c => collectFilesFromEntry(c, subPath)));
        return nested.flat();
    }
    return [];
}

// ── Styles ────────────────────────────────────────────────────────────────────

const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid #e2e8f0',
    borderRadius: '0.5rem',
    overflow: 'hidden',
    backgroundColor: '#fff',
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    height: '100%',
    minHeight: 0,
    position: 'relative',  // needed for drop overlay
};

const headerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '0.5rem',
    padding: '0.5rem 0.75rem',
    backgroundColor: '#1e293b',
    color: '#f1f5f9',
    flexShrink: 0,
};

const breadcrumbStyle = {
    fontFamily: 'monospace',
    fontSize: '0.9rem',
    fontWeight: 600,
    color: '#ffffff',
};

const toolbarStyle = {
    display: 'flex',
    gap: '0.375rem',
    flexWrap: 'wrap',
};

const tbtnStyle = {
    background: '#334155',
    border: '1px solid #475569',
    color: '#f1f5f9',
    borderRadius: '0.25rem',
    padding: '0.25rem 0.6rem',
    cursor: 'pointer',
    fontSize: '0.8rem',
    whiteSpace: 'nowrap',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.35rem',
};

const tableWrapStyle = {
    overflowX: 'auto',
    overflowY: 'auto',
    flex: 1,
    minHeight: 0,
};

const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.85rem',
};

const thStyle = (active, align = 'left') => ({
    position: 'sticky',
    top: 0,
    backgroundColor: '#f1f5f9',
    padding: '0.4rem 0.6rem',
    textAlign: align,
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    userSelect: 'none',
    fontWeight: 600,
    fontSize: '0.8rem',
    color: active ? '#2563eb' : '#374151',
    borderBottom: '2px solid #e2e8f0',
});

const trStyle = (isDir, hover) => ({
    backgroundColor: hover ? '#eff6ff' : 'transparent',
    borderBottom: '1px solid #f1f5f9',
    cursor: isDir ? 'pointer' : 'default',
});

const tdStyle      = { padding: '0.35rem 0.6rem', whiteSpace: 'nowrap' };
const tdRStyle     = { ...tdStyle, textAlign: 'right' };
const tdNameStyle  = { ...tdStyle, width: '100%', maxWidth: 0, minWidth: '9rem', overflow: 'hidden' };
const tdCheckStyle = { padding: '0.35rem 0.3rem 0.35rem 0.6rem', width: 28, verticalAlign: 'middle' };

const actionBtnStyle = {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    padding: '0.15rem',
    borderRadius: '0.2rem',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    lineHeight: 1,
};

const inputStyle = {
    width: '100%',
    padding: '0.45rem 0.6rem',
    border: '1px solid #cbd5e1',
    borderRadius: '0.375rem',
    fontSize: '0.9rem',
    boxSizing: 'border-box',
};

const labelStyle = {
    display: 'block',
    marginBottom: '0.25rem',
    fontWeight: 500,
    fontSize: '0.85rem',
    color: '#374151',
};

const inlineLabelStyle = {
    fontWeight: 500,
    fontSize: '0.875rem',
    color: '#374151',
    whiteSpace: 'nowrap',
    minWidth: '120px',
    width: '120px',
};

const topLabelStyle = {
    fontWeight: 500,
    fontSize: '0.875rem',
    color: '#374151',
};


// ── Validation helpers ────────────────────────────────────────────────────────

function validateFileOrFolderName(name) {
    if (!name || !name.trim()) return 'Name is required.';
    if (!isValidFileOrFolderName(name)) return 'Name contains invalid characters or ends with a space/dot. Characters \\ / : * ? " < > | and control characters are not allowed.';
    return '';
}

function isJsxFile(item) {
    return !!item
        && !item.isDir
        && typeof item.name === 'string'
        && item.name.toLowerCase().endsWith('.jsx');
}

// ── Context menu ──────────────────────────────────────────────────────────────

function buildCtxMenuItems({ item, canModify, canCreateOrUpload, isMarkdown, onPreview, onMarkdownPreview, onJsxPreview, onDownload, onRename, onMove, onDelete, onNewFolder, onNewFile, onNewExternalSource, extraForFile, extraForFolder }) {
    const canEdit    = item.isMarkdown || item.isText;
    const canPreview = item.isImage;
    const canViewJsx = !!onJsxPreview && isJsxFile(item);
    const items = [];

    if (canEdit)    items.push({ label: 'Edit',          icon: <Ic d={ICONS.search}   fill="#2563eb" size={16} />, color: '#2563eb', onClick: () => onPreview(item) });
    if (canViewJsx) items.push({ label: 'View JSX',      icon: <Ic d={ICONS.view}     fill="#2563eb" size={16} />, color: '#2563eb', onClick: () => onJsxPreview(item) });
    if (canPreview) items.push({ label: 'Preview',       icon: <Ic d={ICONS.search}   fill="#2563eb" size={16} />, color: '#2563eb', onClick: () => onPreview(item) });
    if (isMarkdown) items.push({ label: 'View Markdown', icon: <Ic d={ICONS.view}     fill="#2563eb" size={16} />, color: '#2563eb', onClick: () => onMarkdownPreview(item) });

    items.push({ label: 'Download', icon: <Ic d={ICONS.download} fill="#374151" size={16} />, onClick: () => onDownload(item) });

    if (canModify) {
        items.push({ type: 'separator' });
        if (canCreateOrUpload) {
            items.push({ label: 'New Folder', icon: <Ic d={ICONS.newFolder} fill="#374151" size={16} />, onClick: onNewFolder });
            items.push({ label: 'New File', icon: <Ic d={ICONS.newFile} fill="#374151" size={16} />, onClick: onNewFile });
            items.push({ label: 'New External Source', icon: <Ic d={ICONS.web} fill="#374151" size={16} />, onClick: onNewExternalSource });
            items.push({ type: 'separator' });
        }
        items.push({ label: 'Rename', icon: <Ic d={ICONS.rename} fill="#374151" size={16} />, onClick: () => onRename(item) });
        items.push({ label: 'Move',   icon: <Ic d={ICONS.move}   fill="#374151" size={16} />, onClick: () => onMove(item)   });
        items.push({ type: 'separator' });
        items.push({ label: 'Delete', icon: <Ic d={ICONS.delete} fill="var(--red_primary)" size={16} />, color: 'var(--red_primary)', onClick: () => onDelete(item) });
    }

    const extraFn = item.isDir ? extraForFolder : extraForFile;
    if (extraFn) {
        const extraItems = extraFn(item);
        if (extraItems && extraItems.length > 0) {
            items.push({ type: 'separator' });
            extraItems.forEach(ei => {
                if (ei.type === 'separator') {
                    items.push(ei);
                } else {
                    items.push({
                        ...ei,
                        onClick: ei.onClick ? () => ei.onClick(item) : undefined,
                    });
                }
            });
        }
    }

    return items;
}

// ── Inline row with hover ─────────────────────────────────────────────────────

function FileRow({ item, checked, onCheck, canModify, onNavigate, onDoubleClick,
                   onMarkdownPreview, onJsxPreview, onDownload, onRename, onMove, onDelete, onContextMenu,
                   folderColor, markdownFileColor, textFileColor, importantFileColor, imageFileColor }) {
    const [hovered, setHovered] = useState(false);
    const canPreview = item.isText || item.isImage;
    const canViewJsx = isJsxFile(item);
    const clickTimerRef = useRef(null);

    const nameColor = item.isDir         ? folderColor
        : item.isMarkdown                ? markdownFileColor
        : item.isText                    ? textFileColor
        : item.isImportant               ? importantFileColor
        : item.isImage                   ? imageFileColor
        : '#374151';
    const iconFill = item.isDir          ? folderColor
        : item.isMarkdown                ? markdownFileColor
        : item.isText                    ? textFileColor
        : item.isImportant               ? importantFileColor
        : item.isImage                   ? imageFileColor
        : 'var(--blue_secondary)';

    const handleNameClick = () => {
        if (item.isDir) { onNavigate(item); return; }
        if (item.isMarkdown || canViewJsx) {
            const handleSpecialPreview = item.isMarkdown ? onMarkdownPreview : onJsxPreview;
            if (clickTimerRef.current) {
                clearTimeout(clickTimerRef.current);
                clickTimerRef.current = null;
                handleSpecialPreview(item);
            } else {
                clickTimerRef.current = setTimeout(() => {
                    clickTimerRef.current = null;
                    onDoubleClick(item);
                }, 280);
            }
        } else {
            onDoubleClick(item);
        }
    };

    return (
        <tr
            style={trStyle(item.isDir, hovered)}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            onContextMenu={e => { e.preventDefault(); e.stopPropagation(); onContextMenu(e, item); }}
        >
            <td style={tdCheckStyle} onClick={e => e.stopPropagation()}>
                <input type="checkbox" checked={checked} onChange={() => onCheck(item.path)}
                    style={{ cursor: 'pointer', accentColor: '#2563eb' }} />
            </td>
            <td style={{ ...tdNameStyle, cursor: 'pointer', color: nameColor }}
                onClick={handleNameClick}
                title={item.name}
            >
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', width: '100%' }}>
                    <Ic d={item.isDir ? ICONS.folder : ICONS.file} fill={iconFill} size={16} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>
                        {item.name}
                    </span>
                </span>
            </td>
            <td style={tdRStyle}>{item.type}</td>
            <td style={tdRStyle}>{formatSize(item.size)}</td>
            <td style={tdRStyle}>{formatDate(item.modified)}</td>
            <td style={{ ...tdRStyle, display: 'flex', gap: '0.15rem', justifyContent: 'flex-end' }}>
                {canViewJsx && (
                    <button style={actionBtnStyle} title="View JSX"
                        onClick={() => onJsxPreview(item)}>
                        <Ic d={ICONS.view} fill="#2563eb" size={18} />
                    </button>
                )}
                {item.isMarkdown && (
                    <button style={actionBtnStyle} title="View Markdown"
                        onClick={() => onMarkdownPreview(item)}>
                        <Ic d={ICONS.view} fill="#2563eb" size={18} />
                    </button>
                )}
                {item.isMarkdown && (
                    <button style={actionBtnStyle} title="Edit"
                        onClick={() => onDoubleClick(item)}>
                        <Ic d={ICONS.search} fill="#2563eb" size={18} />
                    </button>
                )}
                {canPreview && (
                    <button style={actionBtnStyle}
                        title={item.isText ? 'Edit' : 'Preview'}
                        onClick={() => onDoubleClick(item)}>
                        <Ic d={ICONS.search} fill="#2563eb" size={18} />
                    </button>
                )}
                <button style={actionBtnStyle} title="Download"
                    onClick={() => onDownload(item)}>
                    <Ic d={ICONS.download} fill="#374151" size={18} />
                </button>
                {canModify && <>
                    <button style={actionBtnStyle} title="Rename"
                        onClick={() => onRename(item)}>
                        <Ic d={ICONS.rename} fill="#374151" size={18} />
                    </button>
                    <button style={actionBtnStyle} title="Move"
                        onClick={() => onMove(item)}>
                        <Ic d={ICONS.move} fill="#374151" size={18} />
                    </button>
                    <button style={actionBtnStyle} title="Delete"
                        onClick={() => onDelete(item)}>
                        <Ic d={ICONS.delete} fill="var(--red_primary)" size={18} />
                    </button>
                </>}
            </td>
        </tr>
    );
}

// ── Column header ─────────────────────────────────────────────────────────────

const COLUMNS = [
    { key: '_check',   label: '',         align: 'left'  },
    { key: 'name',     label: 'Name',     align: 'left'  },
    { key: 'type',     label: 'Type',     align: 'right' },
    { key: 'size',     label: 'Size',     align: 'right' },
    { key: 'modified', label: 'Modified', align: 'right' },
    { key: '_actions', label: '',         align: 'right' },
];

// ── Main component ────────────────────────────────────────────────────────────

const FolderBrowser = ({ rootFolder, mode: modeProp, onSelectionChange, contextMenuExtraForFile, contextMenuExtraForFolder }) => {
    const mode = modeProp ?? 'admin';
    const canCreateOrUpload = mode === 'admin';
    const canModify = mode === 'admin' || mode === 'uploadAndCreateDenied';

    // Navigation
    const [currentPath, setCurrentPath] = useState('');

    // List data
    const [items, setItems] = useState([]);
    const [total, setTotal] = useState(0);
    const [hasMore, setHasMore] = useState(false);
    const [remaining, setRemaining] = useState(0);
    const [loadedCount, setLoadedCount] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [listError, setListError] = useState('');

    // Sorting
    const [sortBy, setSortBy] = useState('name');
    const [sortDir, setSortDir] = useState('asc');

    // Modals — null = closed, object = open with data
    const [uploadModal, setUploadModal] = useState(null);
    const [newFolderModal, setNewFolderModal] = useState(null);
    const [newTextModal, setNewTextModal] = useState(null);
    const [weblinkModal, setWeblinkModal] = useState(null);
    const [weblinkOverwriteConfirm, setWeblinkOverwriteConfirm] = useState(false);
    const [newMenuOpen, setNewMenuOpen] = useState(false);
    const newBtnRef = useRef(null);
    const [bulkActionsMenuOpen, setBulkActionsMenuOpen] = useState(false);
    const bulkActionsBtnRef = useRef(null);
    const [renameModal, setRenameModal] = useState(null);
    const [moveModal, setMoveModal] = useState(null);
    const [deleteModal, setDeleteModal] = useState(null);
    const [editorModal, setEditorModal] = useState(null);
    const [autoSaveSecondsLeft, setAutoSaveSecondsLeft] = useState(null);
    const editorModalRef = useRef(null);
    editorModalRef.current = editorModal;
    const [closeEditorConfirmModal, setCloseEditorConfirmModal] = useState(false);
    const [imageModal, setImageModal] = useState(null);
    const [ctxMenu, setCtxMenu] = useState(null); // { item, x, y }

    // Multi-selection
    const [selected, setSelected] = useState(new Set()); // Set of item.path strings
    const [bulkMoveModal,     setBulkMoveModal]     = useState(null); // { items, destPath, error }
    const [bulkDeleteModal,   setBulkDeleteModal]   = useState(null); // { items, error }
    const [bulkDownloadModal, setBulkDownloadModal] = useState(null); // { items }

    useEffect(() => { onSelectionChange?.(selected); }, [selected, onSelectionChange]);
    useEffect(() => { setSelected(new Set()); }, [rootFolder]);

    // Upload conflict resolution
    // { file, filename, pendingFiles, applyAll, errors, itemsSnapshot, fileNum, fileTotal }
    const [conflictModal, setConflictModal] = useState(null);

    // Folder conflict — shown when dropped folders match existing directories
    // { folderNames: string[], allFiles: [{file,relativePath}], itemsSnapshot }
    const [folderConflictModal, setFolderConflictModal] = useState(null);

    // Upload progress modal — null = hidden
    // { filename, fileNum, fileTotal, percent }
    const [uploadProgress, setUploadProgress] = useState(null);
    const abortRef = useRef(null);

    // Editor config fetched once from the backend (single source of truth)
    const [editorConfig, setEditorConfig] = useState({
        folderColor: '#a16207',
        markdownFileColor: '#007700',
        markdownFileExtensions: [],
        textFileColor: '#2563eb',
        textExtensions: [],
        importantFileColor: 'var(--red_primary)',
        importantFileExtensions: [],
        imageFileColor: '#2563eb',
        imageExtensions: [],
    });
    useEffect(() => {
        apiFetch('/api/folder/config', {})
            .then(data => setEditorConfig(data))
            .catch(() => {});
    }, []);

    /** Resolve the CodeEditor grammar for a filename using the fetched config. */
    const grammarFor = useCallback((filename) => {
        const ext = filename?.includes('.')
            ? '.' + filename.split('.').pop().toLowerCase()
            : '';
        return (
            editorConfig.markdownFileExtensions.find(e => e.extension === ext)?.grammar
            ?? editorConfig.textExtensions.find(e => e.extension === ext)?.grammar
            ?? 'plaintext'
        );
    }, [editorConfig]);

    // ── Load list ─────────────────────────────────────────────────────────────

    const loadItems = useCallback(async (path, skip = 0, append = false) => {
        setIsLoading(true);
        setListError('');
        try {
            const data = await apiFetch('/api/folder/list', {
                rootFolder, path, skip, limit: INITIAL_LIMIT,
            });
            setItems(prev => append ? [...prev, ...data.items] : data.items);
            setTotal(data.total);
            setHasMore(data.hasMore);
            setRemaining(data.remaining);
            setLoadedCount(skip + data.items.length);
        } catch (e) {
            setListError(e.message);
        } finally {
            setIsLoading(false);
        }
    }, [rootFolder]);

    useEffect(() => { loadItems(currentPath); }, [currentPath, loadItems]);

    const refresh = useCallback(() => { setSelected(new Set()); loadItems(currentPath); }, [currentPath, loadItems]);

    // ── Upload conflict helpers ───────────────────────────────────────────────

    /** Upload a single file using XHR so progress and cancellation work. */
    const doUpload = async (filename, file, uploadPath) => {
        const controller = new AbortController();
        abortRef.current = controller;
        const fd = new FormData();
        fd.append('rootFolder', rootFolder);
        fd.append('path', uploadPath ?? currentPath);
        fd.append('filename', filename);
        fd.append('file', file);
        return apiUploadWithProgress('/api/folder/upload', fd, {
            onProgress: (pct) => setUploadProgress(p => p ? { ...p, percent: pct } : p),
            signal: controller.signal,
        });
    };

    /**
     * Upload a sequence of file entries, pausing when a conflict is found.
     * Each entry is { file, relativePath } where relativePath is the subfolder
     * relative to currentPath (empty string for files dropped directly).
     * fileNumStart and fileTotal track position in the overall batch for the progress label.
     * itemsSnapshot is the directory listing captured at the start of the operation.
     */
    const processUploadQueue = async (entries, overwriteAll, errors, itemsSnapshot, fileNumStart = 1, fileTotal = null) => {
        const total = fileTotal ?? entries.length;
        for (let idx = 0; idx < entries.length; idx++) {
            const entry = entries[idx];
            const file = entry.file;
            const filename = file.name;
            const relativePath = entry.relativePath || '';
            const uploadPath = relativePath
                ? (currentPath ? `${currentPath}/${relativePath}` : relativePath)
                : currentPath;
            const fileNum = fileNumStart + idx;
            const displayName = relativePath ? `${relativePath}/${filename}` : filename;

            if (!relativePath && !overwriteAll && itemsSnapshot.some(i => !i.isDir && i.name === filename)) {
                setUploadProgress(null);
                setConflictModal({
                    file, filename, relativePath,
                    pendingFiles: entries.slice(idx + 1),
                    applyAll: false, errors, itemsSnapshot,
                    fileNum, fileTotal: total,
                });
                return;
            }

            setUploadProgress({ filename: displayName, fileNum, fileTotal: total, percent: 0 });
            try {
                await doUpload(filename, file, uploadPath);
            } catch (err) {
                if (err.name === 'AbortError') { setUploadProgress(null); return; }
                errors = [...errors, `${displayName}: ${err.message}`];
            }
        }
        setUploadProgress(null);
        if (errors.length) setListError(errors.join(' • '));
        else refresh();
    };

    const handleConflictOverwrite = async () => {
        const { file, filename, relativePath = '', pendingFiles, applyAll, errors, itemsSnapshot, fileNum, fileTotal } = conflictModal;
        const fn  = fileNum  ?? 1;
        const ft  = fileTotal ?? (1 + pendingFiles.length);
        const uploadPath = relativePath
            ? (currentPath ? `${currentPath}/${relativePath}` : relativePath)
            : currentPath;
        setConflictModal(null);
        setUploadProgress({ filename, fileNum: fn, fileTotal: ft, percent: 0 });
        let updatedErrors = [...errors];
        try {
            await doUpload(filename, file, uploadPath);
        } catch (err) {
            if (err.name === 'AbortError') { setUploadProgress(null); return; }
            updatedErrors = [...updatedErrors, `${filename}: ${err.message}`];
        }
        if (pendingFiles.length === 0) {
            setUploadProgress(null);
            if (updatedErrors.length) setListError(updatedErrors.join(' • '));
            else refresh();
        } else {
            await processUploadQueue(pendingFiles, applyAll, updatedErrors, itemsSnapshot, fn + 1, ft);
        }
    };

    const handleConflictSkip = () => {
        const { pendingFiles, applyAll, errors, itemsSnapshot, fileNum, fileTotal } = conflictModal;
        const fn = fileNum  ?? 1;
        const ft = fileTotal ?? (1 + pendingFiles.length);
        setConflictModal(null);
        if (pendingFiles.length === 0) {
            if (errors.length) setListError(errors.join(' • '));
            else refresh();
        } else {
            processUploadQueue(pendingFiles, applyAll, errors, itemsSnapshot, fn + 1, ft);
        }
    };

    const handleCancelUpload = () => {
        abortRef.current?.abort();
        setUploadProgress(null);
    };

    // ── Drag-and-drop upload ──────────────────────────────────────────────────

    const [isDragOver, setIsDragOver] = useState(false);
    const dragCountRef = useRef(0);

    const handleDragEnter = useCallback((e) => {
        e.preventDefault();
        if (canCreateOrUpload && [...e.dataTransfer.types].includes('Files')) {
            dragCountRef.current += 1;
            setIsDragOver(true);
        }
    }, [canCreateOrUpload]);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        dragCountRef.current -= 1;
        if (dragCountRef.current === 0) setIsDragOver(false);
    }, []);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }, []);

    const handleDrop = useCallback(async (e) => {
        e.preventDefault();
        dragCountRef.current = 0;
        setIsDragOver(false);
        if (!canCreateOrUpload) return;

        let allFiles = [];
        const dtItems = e.dataTransfer.items;
        if (dtItems && dtItems.length) {
            const entries = [...dtItems]
                .map(i => i.webkitGetAsEntry?.())
                .filter(Boolean);
            if (entries.length > 0) {
                for (const entry of entries) {
                    const collected = await collectFilesFromEntry(entry);
                    allFiles = allFiles.concat(collected);
                }
            }
        }
        if (allFiles.length === 0) {
            allFiles = Array.from(e.dataTransfer.files)
                .map(f => ({ file: f, relativePath: '' }));
        }

        if (allFiles.length === 0) return;
        if (allFiles.length === 1 && !allFiles[0].relativePath) {
            setUploadModal({ file: allFiles[0].file, filename: allFiles[0].file.name, error: '', uploading: false });
            return;
        }

        setListError('');

        const topLevelFolders = [...new Set(
            allFiles.filter(f => f.relativePath).map(f => f.relativePath.split('/')[0])
        )];
        const conflicting = topLevelFolders.filter(
            name => items.some(i => i.isDir && i.name === name)
        );

        if (conflicting.length > 0) {
            setFolderConflictModal({ folderNames: conflicting, allFiles, itemsSnapshot: items });
            return;
        }

        await processUploadQueue(allFiles, false, [], items, 1, allFiles.length);
    }, [canCreateOrUpload, currentPath, rootFolder, refresh, items]); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Sorting ───────────────────────────────────────────────────────────────

    const sortedItems = useMemo(() => sortItems(items, sortBy, sortDir), [items, sortBy, sortDir]);

    const handleSort = (col) => {
        if (col === '_actions' || col === '_check') return;
        setSortDir(prev => col === sortBy ? (prev === 'asc' ? 'desc' : 'asc') : 'asc');
        setSortBy(col);
    };

    // ── Navigation ────────────────────────────────────────────────────────────

    const handleNavigate = (item) => {
        setItems([]);
        setSelected(new Set());
        setCurrentPath(item.path);
    };

    const handleNavigateUp = () => {
        const parts = currentPath.split('/').filter(Boolean);
        parts.pop();
        setItems([]);
        setSelected(new Set());
        setCurrentPath(parts.join('/'));
    };

    // ── Download ──────────────────────────────────────────────────────────────
    // For files: single-file download. For folders: same as Bulk Action -> Bulk Download (that folder only).

    const handleDownload = (item) => {
        if (item.isDir) {
            setBulkDownloadModal({ items: [item] });
            return;
        }
        (async () => {
            try {
                await apiDownload('/api/folder/download', { rootFolder, path: currentPath, name: item.name }, item.name);
            } catch (e) {
                setListError(e.message);
            }
        })();
    };

    // ── Double-click (text editor / image preview) ────────────────────────────

    const handleDoubleClick = async (item) => {
        if (item.isDir) { handleNavigate(item); return; }
        if (item.isMarkdown || item.isText) {
            try {
                const data = await apiFetch('/api/folder/read', { rootFolder, path: currentPath, name: item.name });
                setEditorModal({ item, content: data.content, isDirty: false });
            } catch (e) { setListError(e.message); }
        } else if (item.isImage) {
            try {
                const url = await apiBlob('/api/folder/download', { rootFolder, path: currentPath, name: item.name });
                setImageModal({ item, url });
            } catch (e) { setListError(e.message); }
        } else {
            handleDownload(item);
        }
    };

    // ── Breadcrumb display ────────────────────────────────────────────────────

    const breadcrumbSegments = [
        { label: rootFolder, path: '' },
        ...currentPath.split('/').filter(Boolean).map((seg, i, arr) => ({
            label: seg,
            path: arr.slice(0, i + 1).join('/'),
        })),
    ];

    // ─────────────────────────────────────────────────────────────────────────
    // UPLOAD MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleUploadOk = async () => {
        const err = validateFileOrFolderName(uploadModal.filename);
        if (err) { setUploadModal(p => ({ ...p, error: err })); return; }
        if (!uploadModal.file) { setUploadModal(p => ({ ...p, error: 'Select a file first.' })); return; }
        // Check for conflict against the current directory listing
        if (items.some(i => !i.isDir && i.name === uploadModal.filename)) {
            setConflictModal({
                file: uploadModal.file,
                filename: uploadModal.filename,
                pendingFiles: [],
                applyAll: false,
                errors: [],
                itemsSnapshot: items,
                fileNum: 1,
                fileTotal: 1,
            });
            setUploadModal(null);
            return;
        }
        setUploadModal(null);
        setUploadProgress({ filename: uploadModal.filename, fileNum: 1, fileTotal: 1, percent: 0 });
        try {
            await doUpload(uploadModal.filename, uploadModal.file);
            setUploadProgress(null);
            refresh();
        } catch (e) {
            setUploadProgress(null);
            if (e.name !== 'AbortError') setListError(e.message);
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // NEW FOLDER MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleNewFolderOk = async () => {
        const err = validateFileOrFolderName(newFolderModal.name);
        if (err) { setNewFolderModal(p => ({ ...p, error: err })); return; }
        try {
            await apiFetch('/api/folder/mkdir', { rootFolder, path: currentPath, name: newFolderModal.name });
            setNewFolderModal(null);
            refresh();
        } catch (e) { setNewFolderModal(p => ({ ...p, error: e.message })); }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // NEW TEXT FILE MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleNewTextOk = async () => {
        const err = validateFileOrFolderName(newTextModal.name);
        if (err) { setNewTextModal(p => ({ ...p, error: err })); return; }
        try {
            await apiFetch('/api/folder/create-text', { rootFolder, path: currentPath, name: newTextModal.name });
            setNewTextModal(null);
            refresh();
        } catch (e) { setNewTextModal(p => ({ ...p, error: e.message })); }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // WEBLINK MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const openExternalSourceModal = useCallback(() => {
        setWeblinkModal({
            name: '', type: 'url',
            url: '', crawlerDepth: 0,
            useCertificate: false, login: '', password: '', thumbprint: '', secret: '',
            fetchCommand: '',
            bookkitAwid: '', authorization: '', loadPageUrl: '',
        });
    }, []);

    const doSaveWeblink = async () => {
        try {
            const body = {
                rootFolder, path: currentPath,
                name: weblinkModal.name,
                type: weblinkModal.type,
                url: weblinkModal.url,
                crawlerDepth: weblinkModal.crawlerDepth,
                useCertificate: weblinkModal.useCertificate,
                login: weblinkModal.login,
                password: weblinkModal.password,
                thumbprint: weblinkModal.thumbprint,
                secret: weblinkModal.secret,
                command: weblinkModal.fetchCommand,
                bookkitAwid: weblinkModal.bookkitAwid,
                authorization: weblinkModal.authorization,
                loadPageUrl: weblinkModal.loadPageUrl,
            };
            await apiFetch('/api/folder/create-weblink', body);
            setWeblinkModal(null);
            refresh();
        } catch (e) { setWeblinkModal(p => ({ ...p, apiError: e.message })); }
    };

    const handleWeblinkOk = async () => {
        const nameErr = validateFileOrFolderName(weblinkModal.name);
        if (nameErr) return;
        const filename = `${weblinkModal.name}.weblink`;
        if (items.some(i => !i.isDir && i.name === filename)) {
            setWeblinkOverwriteConfirm(true);
            return;
        }
        await doSaveWeblink();
    };

    // ─────────────────────────────────────────────────────────────────────────
    // RENAME MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleRenameOk = async () => {
        const err = validateFileOrFolderName(renameModal.newName);
        if (err) { setRenameModal(p => ({ ...p, error: err })); return; }
        try {
            await apiFetch('/api/folder/rename', {
                rootFolder, path: currentPath, name: renameModal.item.name, newName: renameModal.newName,
            });
            setRenameModal(null);
            refresh();
        } catch (e) { setRenameModal(p => ({ ...p, error: e.message })); }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // MOVE MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleMoveOk = async () => {
        const srcPath = currentPath
            ? `${currentPath}/${moveModal.item.name}`
            : moveModal.item.name;
        try {
            await apiFetch('/api/folder/move', {
                rootFolder, srcPath, dstPath: moveModal.destPath,
            });
            setMoveModal(null);
            refresh();
        } catch (e) { setMoveModal(p => ({ ...p, error: e.message })); }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // DELETE MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleDeleteOk = async () => {
        try {
            await apiFetch('/api/folder/delete', {
                rootFolder, path: currentPath, name: deleteModal.item.name,
            });
            setDeleteModal(null);
            refresh();
        } catch (e) { setDeleteModal(p => ({ ...p, error: e.message })); }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // TEXT EDITOR MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleEditorSave = async () => {
        try {
            await apiFetch('/api/folder/write', {
                rootFolder, path: currentPath, name: editorModal.item.name, content: editorModal.content,
            });
            setEditorModal(p => ({ ...p, isDirty: false }));
            refresh();
        } catch (e) { setEditorModal(p => ({ ...p, saveError: e.message })); }
    };

    const handleEditorSaveAndExit = async () => {
        try {
            await apiFetch('/api/folder/write', {
                rootFolder, path: currentPath, name: editorModal.item.name, content: editorModal.content,
            });
            setEditorModal(null);
            refresh();
        } catch (e) { setEditorModal(p => ({ ...p, saveError: e.message })); }
    };

    const handleEditorSaveRef = useRef(handleEditorSave);
    handleEditorSaveRef.current = handleEditorSave;

    // Ctrl+S in editor modal triggers Save (only when modal is open)
    useEffect(() => {
        if (!editorModal) return;
        const onKeyDown = (e) => {
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                if (editorModal?.isDirty && handleEditorSaveRef.current) handleEditorSaveRef.current();
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [editorModal]);

    // Auto-save every 60 seconds while there are unsaved changes; shows countdown in last 5 s
    useEffect(() => {
        if (!editorModal?.isDirty || !canModify) {
            setAutoSaveSecondsLeft(null);
            return;
        }
        setAutoSaveSecondsLeft(60);
        const id = setInterval(() => {
            setAutoSaveSecondsLeft(prev => {
                if (prev <= 1) {
                    handleEditorSaveRef.current?.();
                    return 60;
                }
                return prev - 1;
            });
        }, 1000);
        return () => clearInterval(id);
    }, [editorModal?.isDirty, editorModal?.item?.path, canModify]); // eslint-disable-line react-hooks/exhaustive-deps

    // ─────────────────────────────────────────────────────────────────────────
    // IMAGE MODAL
    // ─────────────────────────────────────────────────────────────────────────

    const handleCloseImage = () => {
        if (imageModal?.url) URL.revokeObjectURL(imageModal.url);
        setImageModal(null);
    };

    /** Close editor modal; if there are unsaved changes, ask for confirmation via modal. */
    const handleCloseEditor = useCallback(() => {
        if (editorModal?.isDirty) {
            setCloseEditorConfirmModal(true);
            return;
        }
        setEditorModal(null);
    }, [editorModal?.isDirty]);


    const imageItems = useMemo(() => sortedItems.filter(i => i.isImage), [sortedItems]);

    const handlePrevImage = useCallback(async () => {
        if (!imageModal) return;
        const idx = imageItems.findIndex(i => i.path === imageModal.item.path);
        if (idx <= 0) return;
        const prevItem = imageItems[idx - 1];
        try {
            if (imageModal.url) URL.revokeObjectURL(imageModal.url);
            const url = await apiBlob('/api/folder/download', { rootFolder, path: currentPath, name: prevItem.name });
            setImageModal({ item: prevItem, url });
        } catch (e) { setListError(e.message); }
    }, [imageModal, imageItems, rootFolder, currentPath]);

    const handleNextImage = useCallback(async () => {
        if (!imageModal) return;
        const idx = imageItems.findIndex(i => i.path === imageModal.item.path);
        if (idx === -1 || idx >= imageItems.length - 1) return;
        const nextItem = imageItems[idx + 1];
        try {
            if (imageModal.url) URL.revokeObjectURL(imageModal.url);
            const url = await apiBlob('/api/folder/download', { rootFolder, path: currentPath, name: nextItem.name });
            setImageModal({ item: nextItem, url });
        } catch (e) { setListError(e.message); }
    }, [imageModal, imageItems, rootFolder, currentPath]);

    // ── Markdown rendered preview (new tab) ───────────────────────────────────

    const handleMarkdownPreview = useCallback((item) => {
        const params = new URLSearchParams({ rootFolder, path: currentPath, name: item.name });
        window.open(`/markdownview?${params}`, '_blank');
    }, [rootFolder, currentPath]);

    const handleJsxPreview = useCallback((item) => {
        const params = new URLSearchParams({ rootFolder, path: currentPath, name: item.name });
        window.open(`/jsxview?${params}`, '_blank');
    }, [rootFolder, currentPath]);

    // ── Selection ─────────────────────────────────────────────────────────────

    const toggleSelect = (path) => {
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(path)) next.delete(path); else next.add(path);
            return next;
        });
    };

    const toggleSelectAll = () => {
        setSelected(prev =>
            prev.size === sortedItems.length
                ? new Set()
                : new Set(sortedItems.map(i => i.path))
        );
    };

    // ── Bulk Move ─────────────────────────────────────────────────────────────

    const handleBulkMoveOk = async () => {
        const errors = [];
        for (const item of bulkMoveModal.items) {
            try {
                await apiFetch('/api/folder/move', {
                    rootFolder, srcPath: item.path, dstPath: bulkMoveModal.destPath,
                });
            } catch (e) {
                errors.push(`${item.name}: ${e.message}`);
            }
        }
        if (errors.length) {
            setBulkMoveModal(p => ({ ...p, error: errors.join(' | ') }));
        } else {
            setBulkMoveModal(null);
            refresh();
        }
    };

    // ── Bulk Delete ───────────────────────────────────────────────────────────

    const handleBulkDeleteOk = async () => {
        const errors = [];
        for (const item of bulkDeleteModal.items) {
            try {
                await apiFetch('/api/folder/delete', {
                    rootFolder, path: currentPath, name: item.name,
                });
            } catch (e) {
                errors.push(`${item.name}: ${e.message}`);
            }
        }
        if (errors.length) {
            setBulkDeleteModal(p => ({ ...p, error: errors.join(' | ') }));
        } else {
            setBulkDeleteModal(null);
            refresh();
        }
    };

    // ── Bulk Download ─────────────────────────────────────────────────────────

    const handleBulkDownloadOk = async () => {
        const { items } = bulkDownloadModal;
        setBulkDownloadModal(null);
        const paths = items.map(i => i.path);
        try {
            await apiDownload('/api/folder/bulk-download', { rootFolder, paths }, 'bulk-download.zip');
        } catch (e) {
            setListError(e.message);
        }
    };

    // ── Context menu ──────────────────────────────────────────────────────────

    const handleContextMenu = (e, item) => {
        setCtxMenu({ item, x: e.clientX, y: e.clientY });
    };


    // ─────────────────────────────────────────────────────────────────────────
    // RENDER
    // ─────────────────────────────────────────────────────────────────────────

    return (
        <div
            style={containerStyle}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            {/* ── Drop overlay ── */}
            {isDragOver && (
                <div style={{
                    position: 'absolute', inset: 0, zIndex: 200,
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    border: '3px dashed #2563eb',
                    borderRadius: '0.5rem',
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    gap: '0.5rem',
                    pointerEvents: 'none',
                }}>
                    <Ic d={ICONS.upload} fill="#2563eb" size={48} />
                    <span style={{ fontSize: '1.25rem', fontWeight: 600, color: '#2563eb' }}>
                        Drop files to upload
                    </span>
                    <span style={{ fontSize: '0.85rem', color: 'var(--blue_primary)' }}>
                        into: /{currentPath || ''}
                    </span>
                </div>
            )}

            {/* ── Header ── */}
            <div style={headerStyle}>
                <span style={{ ...breadcrumbStyle, display: 'inline-flex', alignItems: 'center', gap: '0.25rem', flexWrap: 'wrap' }}>
                    <Ic d={ICONS.folder} fill="#ffffff" size={18} />
                    {breadcrumbSegments.map((seg, i) => {
                        const isLast = i === breadcrumbSegments.length - 1;
                        return (
                            <React.Fragment key={seg.path + i}>
                                {i > 0 && <span style={{ color: 'var(--blue_secondary_disabled)', userSelect: 'none' }}>›</span>}
                                {isLast ? (
                                    <span>{seg.label}</span>
                                ) : (
                                    <span
                                        style={{ cursor: 'pointer', textDecoration: 'underline', textUnderlineOffset: '2px' }}
                                        onClick={() => setCurrentPath(seg.path)}
                                    >
                                        {seg.label}
                                    </span>
                                )}
                            </React.Fragment>
                        );
                    })}
                </span>
                <div style={toolbarStyle}>
                    <button style={tbtnStyle} title="Refresh" onClick={refresh}>
                        <Ic d={ICONS.refresh} fill="#f1f5f9" size={16} /> Refresh
                    </button>
                    <div>
                        <button
                            ref={bulkActionsBtnRef}
                            style={{
                                ...tbtnStyle,
                                opacity: selected.size === 0 ? 0.45 : 1,
                                cursor: selected.size === 0 ? 'not-allowed' : 'pointer',
                            }}
                            title="Bulk Actions"
                            disabled={selected.size === 0}
                            onClick={() => setBulkActionsMenuOpen(o => !o)}
                        >
                            <Ic d={ICONS.moreVert} fill="#f1f5f9" size={16} /> Bulk Actions
                        </button>
                        {bulkActionsMenuOpen && selected.size > 0 && (() => {
                            const selectedItems = sortedItems.filter(i => selected.has(i.path));
                            return (
                                <PopupMenu
                                    anchor={bulkActionsBtnRef.current}
                                    onClose={() => setBulkActionsMenuOpen(false)}
                                    minWidth={190}
                                    items={[
                                        { label: 'Bulk Download', icon: <Ic d={ICONS.download} fill="#374151" size={16} />, onClick: () => setBulkDownloadModal({ items: selectedItems }) },
                                        ...(canModify ? [
                                            { label: 'Bulk Move',   icon: <Ic d={ICONS.move}   fill="#374151" size={16} />, onClick: () => setBulkMoveModal({ items: selectedItems, destPath: '', error: '' }) },
                                            { label: 'Bulk Delete', icon: <Ic d={ICONS.delete} fill="var(--red_primary)" size={16} />, color: 'var(--red_primary)', onClick: () => setBulkDeleteModal({ items: selectedItems, error: '' }) },
                                        ] : []),
                                    ]}
                                />
                            );
                        })()}
                    </div>
                    {canCreateOrUpload && <div>
                        <button
                            ref={newBtnRef}
                            style={tbtnStyle}
                            title="New…"
                            onClick={() => setNewMenuOpen(o => !o)}
                        >
                            <span style={{ fontSize: '1rem', lineHeight: 1, marginRight: '0.1rem' }}>+</span> New
                        </button>
                        {newMenuOpen && (
                            <PopupMenu
                                anchor={newBtnRef.current}
                                onClose={() => setNewMenuOpen(false)}
                                minWidth={190}
                                items={[
                                    { label: 'Upload',          icon: <Ic d={ICONS.upload}    fill="#374151" size={16} />, onClick: () => setUploadModal({ file: null, filename: '', error: '', uploading: false }) },
                                    { type: 'separator' },
                                    { label: 'New Folder',      icon: <Ic d={ICONS.newFolder} fill="#374151" size={16} />, onClick: () => setNewFolderModal({ name: '', error: '' }) },
                                    { label: 'New File',        icon: <Ic d={ICONS.newFile}   fill="#374151" size={16} />, onClick: () => setNewTextModal({ name: '', error: '' }) },
                                    { label: 'External Source', icon: <Ic d={ICONS.web}       fill="#374151" size={16} />, onClick: openExternalSourceModal },
                                ]}
                            />
                        )}
                    </div>}
                </div>
            </div>

            {/* ── Status / Error bar ── */}
            {(listError || isLoading) && (
                <div style={{
                    padding: '0.3rem 0.75rem',
                    fontSize: '0.8rem',
                    backgroundColor: listError ? '#fef2f2' : '#f0f9ff',
                    color: listError ? 'var(--red_primary)' : '#0369a1',
                    flexShrink: 0,
                }}>
                    {isLoading ? 'Loading…' : listError}
                </div>
            )}

            {/* ── File table ── */}
            <div style={tableWrapStyle}>
                <table style={tableStyle}>
                    <thead>
                        <tr>
                            {COLUMNS.map(col => {
                                if (col.key === '_check') {
                                    const allChecked  = sortedItems.length > 0 && selected.size === sortedItems.length;
                                    const someChecked = selected.size > 0 && selected.size < sortedItems.length;
                                    return (
                                        <th key="_check" style={{ ...thStyle(false, 'left'), ...tdCheckStyle, cursor: 'default' }}>
                                            <input type="checkbox" checked={allChecked}
                                                ref={el => el && (el.indeterminate = someChecked)}
                                                onChange={toggleSelectAll}
                                                style={{ cursor: 'pointer', accentColor: '#2563eb' }} />
                                        </th>
                                    );
                                }
                                return (
                                    <th key={col.key}
                                        style={thStyle(sortBy === col.key, col.align)}
                                        onClick={() => handleSort(col.key)}
                                    >
                                        {col.label}
                                        {sortBy === col.key && col.key !== '_actions'
                                            ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                                    </th>
                                );
                            })}
                        </tr>
                    </thead>
                    <tbody>
                        {/* ".." row */}
                        {currentPath && (
                            <tr style={trStyle(true, false)} onClick={handleNavigateUp}>
                                <td style={tdCheckStyle} />
                                <td style={{ ...tdNameStyle, color: editorConfig.folderColor }} colSpan={4}>
                                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                                        <Ic d={ICONS.folder} fill={editorConfig.folderColor} size={16} /> ..
                                    </span>
                                </td>
                                <td style={tdStyle} />
                            </tr>
                        )}
                        {sortedItems.map(item => (
                            <FileRow
                                key={item.path}
                                item={item}
                                checked={selected.has(item.path)}
                                onCheck={toggleSelect}
                                canModify={canModify}
                                onNavigate={handleNavigate}
                                onDoubleClick={handleDoubleClick}
                                onDownload={handleDownload}
                                onRename={i => setRenameModal({ item: i, newName: i.name, error: '' })}
                                onMove={i => setMoveModal({ item: i, destPath: '', error: '' })}
                                onDelete={i => setDeleteModal({ item: i, error: '' })}
                                onMarkdownPreview={handleMarkdownPreview}
                                onJsxPreview={handleJsxPreview}
                                onContextMenu={handleContextMenu}
                                folderColor={editorConfig.folderColor}
                                markdownFileColor={editorConfig.markdownFileColor}
                                textFileColor={editorConfig.textFileColor}
                                importantFileColor={editorConfig.importantFileColor}
                                imageFileColor={editorConfig.imageFileColor}
                            />
                        ))}
                    </tbody>
                </table>

                {/* Load more */}
                {hasMore && (
                    <div style={{ padding: '0.5rem 0.75rem' }}>
                        <Button
                            label={`Load ${remaining} more items`}
                            onClick={() => loadItems(currentPath, loadedCount, true)}
                            color="#475569"
                        />
                    </div>
                )}

                {/* Empty state */}
                {!isLoading && items.length === 0 && !listError && (
                    <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--blue_secondary_disabled)', fontSize: '0.9rem' }}>
                        Folder is empty.
                    </div>
                )}
            </div>

            {/* ══════════════════════════════════════════════════════════════
                MODALS
            ══════════════════════════════════════════════════════════════ */}

            {/* ── Upload ── */}
            <ModalWindow
                isOpen={!!uploadModal}
                title="Upload File"
                onOk={handleUploadOk}
                onCancel={() => setUploadModal(null)}
                okLabel="Upload"
                okDisabled={!uploadModal?.filename || !uploadModal?.file}
                validationErrors={uploadModal?.error ? [uploadModal.error] : []}
            >
                {uploadModal && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                            <label style={labelStyle}>File</label>
                            <input type="file" style={{ fontSize: '0.9rem' }}
                                onChange={e => {
                                    const f = e.target.files[0];
                                    if (!f) return;
                                    setUploadModal(p => ({ ...p, file: f, filename: f.name, error: '' }));
                                }}
                            />
                        </div>
                        <div>
                            <label style={labelStyle} htmlFor="upload-filename">File name on server</label>
                            <input id="upload-filename" style={inputStyle} value={uploadModal.filename}
                                onChange={e => setUploadModal(p => ({ ...p, filename: e.target.value, error: '' }))}
                            />
                        </div>
                    </div>
                )}
            </ModalWindow>

            {/* ── New Folder ── */}
            <ModalWindow
                isOpen={!!newFolderModal}
                title="New Folder"
                onOk={handleNewFolderOk}
                onCancel={() => setNewFolderModal(null)}
                okDisabled={!newFolderModal?.name}
                validationErrors={newFolderModal?.error ? [newFolderModal.error] : []}
            >
                {newFolderModal && (
                    <div>
                        <label style={labelStyle} htmlFor="new-folder-name">Folder name</label>
                        <input id="new-folder-name" style={inputStyle} autoFocus
                            value={newFolderModal.name}
                            onChange={e => setNewFolderModal(p => ({ ...p, name: e.target.value, error: '' }))}
                            onKeyDown={e => e.key === 'Enter' && handleNewFolderOk()}
                        />
                    </div>
                )}
            </ModalWindow>

            {/* ── New File ── */}
            <ModalWindow
                isOpen={!!newTextModal}
                title="New File"
                onOk={handleNewTextOk}
                onCancel={() => setNewTextModal(null)}
                okDisabled={!newTextModal?.name}
                validationErrors={newTextModal?.error ? [newTextModal.error] : []}
            >
                {newTextModal && (
                    <div>
                        <label style={labelStyle} htmlFor="new-text-name">File name (e.g. notes.txt)</label>
                        <input id="new-text-name" style={inputStyle} autoFocus
                            value={newTextModal.name}
                            onChange={e => setNewTextModal(p => ({ ...p, name: e.target.value, error: '' }))}
                            onKeyDown={e => e.key === 'Enter' && handleNewTextOk()}
                        />
                    </div>
                )}
            </ModalWindow>

            {/* ── External Source ── */}
            {weblinkModal && (() => {
                const nameErr = weblinkModal.name ? validateFileOrFolderName(weblinkModal.name) : '';
                const needsUrl = weblinkModal.type === 'url' || weblinkModal.type === 'sharepoint' || weblinkModal.type === 'bookkit';
                const validationErrors = [nameErr, weblinkModal.apiError].filter(Boolean);
                const okDisabled = !weblinkModal.name || (needsUrl && !weblinkModal.url);
                return (
                    <ModalWindow
                        isOpen
                        title="External Source"
                        okLabel="Save"
                        onOk={handleWeblinkOk}
                        onCancel={() => setWeblinkModal(null)}
                        okDisabled={okDisabled}
                        validationErrors={validationErrors}
                        style={{ minWidth: '420px' }}
                        resizable
                    >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div className="responsive-input-container">
                                <label style={inlineLabelStyle} htmlFor="weblink-name">Name:</label>
                                <input id="weblink-name" style={{ ...inputStyle, flex: 1, minWidth: 0 }} autoFocus
                                    value={weblinkModal.name}
                                    onChange={e => setWeblinkModal(p => ({ ...p, name: e.target.value, apiError: '' }))}
                                />
                            </div>
                            <div className="responsive-input-container">
                                <label style={inlineLabelStyle}>Type:</label>
                                <div style={{ flex: 1 }}>
                                    <MultiSwitch
                                        options={[
                                            { value: 'url', label: 'URL' },
                                            { value: 'bookkit', label: 'BookKit' },
                                            { value: 'sharepoint', label: 'SharePoint' },
                                            { value: 'fetch', label: 'Fetch' },
                                        ]}
                                        value={weblinkModal.type}
                                        onChange={v => setWeblinkModal(p => ({ ...p, type: v }))}
                                        selectedColor="var(--blue_primary)"
                                        selectedTextColor="#ffffff"
                                        unselectedColor="#e2e8f0"
                                        textColor="#1e293b"
                                    />
                                </div>
                            </div>
                            <HorizontalSeparator label="Configuration" fullWidth bleed="1rem" color="var(--blue_primary)" />
                            {weblinkModal.type === 'url' && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    <div className="responsive-input-container">
                                        <label style={inlineLabelStyle} htmlFor="weblink-url">URL:</label>
                                        <input id="weblink-url" style={{ ...inputStyle, flex: 1, minWidth: 0 }} placeholder="https://…"
                                            value={weblinkModal.url}
                                            onChange={e => setWeblinkModal(p => ({ ...p, url: e.target.value, apiError: '' }))}
                                        />
                                    </div>
                                    <Slider
                                        label="Crawler Depth:"
                                        labelWidth="120px"
                                        min={0} max={10}
                                        value={weblinkModal.crawlerDepth}
                                        onChange={v => setWeblinkModal(p => ({ ...p, crawlerDepth: v }))}
                                        allowManualInput
                                    />
                                </div>
                            )}
                            {weblinkModal.type === 'bookkit' && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    <div className="responsive-input-container top-label">
                                        <label style={topLabelStyle} htmlFor="weblink-bookkit-awid">BookKit awid:</label>
                                        <input id="weblink-bookkit-awid" style={{ ...inputStyle, width: '100%' }} placeholder="(optional)"
                                            value={weblinkModal.bookkitAwid}
                                            onChange={e => {
                                                const awid = e.target.value;
                                                setWeblinkModal(p => ({
                                                    ...p,
                                                    bookkitAwid: awid,
                                                    url: awid ? `https://uuapp.plus4u.net/uu-bookkit-maing01/${awid}/getBookStructure` : '',
                                                    loadPageUrl: awid ? `https://uuapp.plus4u.net/uu-bookkit-maing01/${awid}/loadPage?code=PAGE_CODE` : '',
                                                }));
                                            }}
                                        />
                                    </div>
                                    <div className="responsive-input-container top-label">
                                        <label style={topLabelStyle} htmlFor="weblink-bookkit-url">URL (getBookStructure cmd):</label>
                                        <input id="weblink-bookkit-url" style={{ ...inputStyle, width: '100%' }} placeholder="https://…"
                                            value={weblinkModal.url}
                                            onChange={e => setWeblinkModal(p => ({ ...p, url: e.target.value, apiError: '' }))}
                                        />
                                    </div>
                                    <div className="responsive-input-container top-label">
                                        <label style={topLabelStyle} htmlFor="weblink-bookkit-load-page-url">Load Page Url:</label>
                                        <input id="weblink-bookkit-load-page-url" style={{ ...inputStyle, width: '100%' }} placeholder="https://…"
                                            value={weblinkModal.loadPageUrl}
                                            onChange={e => setWeblinkModal(p => ({ ...p, loadPageUrl: e.target.value }))}
                                        />
                                    </div>
                                    <div className="responsive-input-container top-label">
                                        <label style={topLabelStyle}>Authorization Token (Bearer):</label>
                                        <CodeEditor
                                            language="plaintext"
                                            showLineNumbers={false}
                                            wordwrap={true}
                                            value={weblinkModal.authorization}
                                            onChange={v => setWeblinkModal(p => ({ ...p, authorization: v }))}
                                            style={{ height: '110px', width: '100%' }}
                                        />
                                    </div>
                                </div>
                            )}
                            {weblinkModal.type === 'sharepoint' && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    <div className="responsive-input-container">
                                        <label style={inlineLabelStyle} htmlFor="weblink-sp-url">URL:</label>
                                        <input id="weblink-sp-url" style={{ ...inputStyle, flex: 1, minWidth: 0 }} placeholder="https://…"
                                            value={weblinkModal.url}
                                            onChange={e => setWeblinkModal(p => ({ ...p, url: e.target.value, apiError: '' }))}
                                        />
                                    </div>
                                    <Switch
                                        label="Use Certificate:"
                                        labelWidth="120px"
                                        value={weblinkModal.useCertificate}
                                        onChange={v => setWeblinkModal(p => ({ ...p, useCertificate: v }))}
                                    />
                                    <div className="responsive-input-container">
                                        <label style={inlineLabelStyle} htmlFor="weblink-login">Login:</label>
                                        <input id="weblink-login" style={{ ...inputStyle, flex: 1, minWidth: 0 }}
                                            value={weblinkModal.login}
                                            onChange={e => setWeblinkModal(p => ({ ...p, login: e.target.value }))}
                                        />
                                    </div>
                                    {!weblinkModal.useCertificate && (
                                        <div className="responsive-input-container">
                                            <label style={inlineLabelStyle} htmlFor="weblink-password">Password:</label>
                                            <input id="weblink-password" style={{ ...inputStyle, flex: 1, minWidth: 0 }}
                                                type="password"
                                                value={weblinkModal.password}
                                                onChange={e => setWeblinkModal(p => ({ ...p, password: e.target.value }))}
                                            />
                                        </div>
                                    )}
                                    {weblinkModal.useCertificate && (
                                        <>
                                            <div className="responsive-input-container">
                                                <label style={inlineLabelStyle} htmlFor="weblink-thumbprint">Thumbprint:</label>
                                                <input id="weblink-thumbprint" style={{ ...inputStyle, flex: 1, minWidth: 0 }}
                                                    value={weblinkModal.thumbprint}
                                                    onChange={e => setWeblinkModal(p => ({ ...p, thumbprint: e.target.value }))}
                                                />
                                            </div>
                                            <div className="responsive-input-container top-label">
                                                <label style={topLabelStyle}>Secret:</label>
                                                <CodeEditor
                                                    language="plaintext"
                                                    showLineNumbers
                                                    wordwrap={false}
                                                    value={weblinkModal.secret}
                                                    onChange={v => setWeblinkModal(p => ({ ...p, secret: v }))}
                                                    style={{ height: '120px', width: '100%' }}
                                                />
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                            {weblinkModal.type === 'fetch' && (
                                <div className="responsive-input-container top-label">
                                    <label style={topLabelStyle}>Fetch Command:</label>
                                    <CodeEditor
                                        language="plaintext"
                                        showLineNumbers
                                        wordwrap={false}
                                        value={weblinkModal.fetchCommand}
                                        onChange={v => setWeblinkModal(p => ({ ...p, fetchCommand: v }))}
                                        style={{ height: '230px', width: '100%' }}
                                    />
                                </div>
                            )}
                        </div>
                    </ModalWindow>
                );
            })()}

            {/* ── Weblink Overwrite Confirm ── */}
            <ModalWindow
                isOpen={weblinkOverwriteConfirm}
                title="File Already Exists"
                okLabel="Overwrite"
                onOk={async () => { setWeblinkOverwriteConfirm(false); await doSaveWeblink(); }}
                onCancel={() => setWeblinkOverwriteConfirm(false)}
                validationErrors={[]}
            >
                <p style={{ margin: 0 }}>
                    File <strong>{weblinkModal?.name}.weblink</strong> already exists in this folder.
                    Do you want to overwrite it?
                </p>
            </ModalWindow>

            {/* ── Rename ── */}
            <ModalWindow
                isOpen={!!renameModal}
                title="Rename"
                onOk={handleRenameOk}
                onCancel={() => setRenameModal(null)}
                okDisabled={!renameModal?.newName}
                validationErrors={renameModal?.error ? [renameModal.error] : []}
            >
                {renameModal && (
                    <div>
                        <label style={labelStyle} htmlFor="rename-input">New name</label>
                        <input id="rename-input" style={inputStyle} autoFocus
                            value={renameModal.newName}
                            onChange={e => setRenameModal(p => ({ ...p, newName: e.target.value, error: '' }))}
                            onKeyDown={e => e.key === 'Enter' && handleRenameOk()}
                        />
                    </div>
                )}
            </ModalWindow>

            {/* ── Move ── */}
            <ModalWindow
                isOpen={!!moveModal}
                title={`Move "${moveModal?.item?.name}"`}
                onOk={handleMoveOk}
                onCancel={() => setMoveModal(null)}
                validationErrors={moveModal?.error ? [moveModal.error] : []}
            >
                {moveModal && (
                    <div>
                        <label style={labelStyle} htmlFor="move-dest">
                            Destination path within <strong>{rootFolder}</strong> (empty = root)
                        </label>
                        <input id="move-dest" style={inputStyle} autoFocus
                            placeholder="e.g. subfolder/nested"
                            value={moveModal.destPath}
                            onChange={e => setMoveModal(p => ({ ...p, destPath: e.target.value, error: '' }))}
                            onKeyDown={e => e.key === 'Enter' && handleMoveOk()}
                        />
                    </div>
                )}
            </ModalWindow>

            {/* ── Delete ── */}
            <ModalWindow
                isOpen={!!deleteModal}
                title="Confirm Delete"
                onOk={handleDeleteOk}
                onCancel={() => setDeleteModal(null)}
                okLabel="Delete"
                okButtonColor="var(--red_primary)"
                validationErrors={deleteModal?.error ? [deleteModal.error] : []}
            >
                {deleteModal && (
                    <p style={{ margin: 0 }}>
                        Delete <strong>{deleteModal.item.name}</strong>?
                        {deleteModal.item.isDir && ' This will delete all contents recursively.'}
                    </p>
                )}
            </ModalWindow>

            {/* ── Text Editor ── */}
            <ModalWindow
                isOpen={!!editorModal}
                title={editorModal ? `Edit — ${editorModal.item.name}${editorModal.isDirty ? ' *' : ''}` : ''}
                onOk={canModify ? handleEditorSaveAndExit : undefined}
                onCancel={handleCloseEditor}
                okLabel="Save and Exit"
                okDisabled={!editorModal?.isDirty}
                cancelLabel="Close"
                validationErrors={editorModal?.saveError ? [editorModal.saveError] : []}
                resizable={true}
                style={{ width: '75vw', height: '65vh' }}
                customFooterButtons={canModify ? [
                    <Button
                        key="save"
                        label={autoSaveSecondsLeft !== null && autoSaveSecondsLeft <= 5 ? `Save (${autoSaveSecondsLeft})` : 'Save'}
                        onClick={handleEditorSave}
                        color={!editorModal?.isDirty ? 'var(--blue_primary_disabled)' : 'var(--blue_primary)'}
                        disabled={!editorModal?.isDirty}
                        style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                        hint="Save File (Ctrl + S)"
                    />,
                ] : []}
            >
                {editorModal && (
                    <CodeEditor
                        value={editorModal.content}
                        readonly={!canModify}
                        language={grammarFor(editorModal.item.name)}
                        onChange={v => setEditorModal(p => ({ ...p, content: v, isDirty: true, saveError: '' }))}
                        title={editorModal.item.name}
                        style={{ height: '100%' }}
                        rootFolder={rootFolder}
                        docPath={currentPath}
                    />
                )}
            </ModalWindow>

            {/* ── Close Editor Without Saving Confirmation ── */}
            <ModalWindow
                isOpen={closeEditorConfirmModal}
                title="Close Without Saving"
                onOk={() => { setCloseEditorConfirmModal(false); setEditorModal(null); }}
                onCancel={() => setCloseEditorConfirmModal(false)}
                okLabel="Close Without Saving"
                okButtonColor="var(--red_primary)"
            >
                <p style={{ margin: 0 }}>Are you sure you want to close without saving? Changes will be lost.</p>
            </ModalWindow>

            {/* ── Image Preview ── */}
            <ModalWindow
                isOpen={!!imageModal}
                title={imageModal?.item?.name ?? ''}
                resizable
                style={{ width: '80vw', height: '80vh' }}
                customFooterButtons={imageModal ? (() => {
                    const idx = imageItems.findIndex(i => i.path === imageModal.item.path);
                    const btnStyle = { height: '40px', display: 'flex', alignItems: 'center' };
                    return [
                        <Button key="prev" label="◀ Previous" color="var(--blue_secondary)"
                            onClick={handlePrevImage}
                            disabled={idx <= 0}
                            style={btnStyle}
                        />,
                        <Button key="next" label="Next ▶" color="var(--blue_secondary)"
                            onClick={handleNextImage}
                            disabled={idx === -1 || idx >= imageItems.length - 1}
                            style={btnStyle}
                        />,
                        <Button key="close" label="Close" color="var(--blue_secondary)"
                            onClick={handleCloseImage}
                            style={btnStyle}
                        />,
                    ];
                })() : []}
            >
                {imageModal && (
                    <Polygon
                        src={imageModal.url}
                        mode="viewer"
                        zoomPanEnabled={true}
                        stretchMode="fit"
                        style={{ width: '100%', height: '100%' }}
                    />
                )}
            </ModalWindow>

            {/* ── Upload Progress ── */}
            <ModalWindow
                isOpen={!!uploadProgress}
                title={uploadProgress?.fileTotal > 1
                    ? `Uploading (${uploadProgress.fileNum} / ${uploadProgress.fileTotal})…`
                    : 'Uploading…'}
                onCancel={handleCancelUpload}
                cancelLabel="Cancel"
            >
                {uploadProgress && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <p style={{ margin: 0, fontSize: '0.9rem', color: '#374151',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {uploadProgress.filename}
                        </p>
                        <ProgressBar value={uploadProgress.percent} />
                    </div>
                )}
            </ModalWindow>

            {/* ── Overwrite Conflict ── */}
            <ModalWindow
                isOpen={!!conflictModal}
                title="File Already Exists"
                onOk={handleConflictOverwrite}
                onCancel={handleConflictSkip}
                okLabel="Overwrite"
                cancelLabel={conflictModal?.pendingFiles?.length > 0 ? 'Skip' : 'Cancel'}
                validationErrors={[]}
            >
                {conflictModal && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <p style={{ margin: 0 }}>
                            File <strong>{conflictModal.filename}</strong> already exists in this
                            folder. Do you want to overwrite it?
                        </p>
                        {conflictModal.pendingFiles.length > 0 && (
                            <label style={{
                                display: 'flex', alignItems: 'center', gap: '0.5rem',
                                cursor: 'pointer', fontSize: '0.9rem', userSelect: 'none',
                            }}>
                                <input
                                    type="checkbox"
                                    checked={conflictModal.applyAll}
                                    onChange={e => setConflictModal(p => ({ ...p, applyAll: e.target.checked }))}
                                    style={{ accentColor: '#2563eb', cursor: 'pointer' }}
                                />
                                Apply to all conflicting files in this batch
                            </label>
                        )}
                    </div>
                )}
            </ModalWindow>

            {/* ── Folder Already Exists ── */}
            <ModalWindow
                isOpen={!!folderConflictModal}
                title="Folder Already Exists"
                onOk={async () => {
                    const { allFiles, itemsSnapshot } = folderConflictModal;
                    setFolderConflictModal(null);
                    await processUploadQueue(allFiles, false, [], itemsSnapshot, 1, allFiles.length);
                }}
                onCancel={() => setFolderConflictModal(null)}
                okLabel="Overwrite"
                cancelLabel="Cancel"
                validationErrors={[]}
            >
                {folderConflictModal && (
                    <div>
                        <p style={{ margin: '0 0 0.5rem' }}>
                            {folderConflictModal.folderNames.length === 1
                                ? <>Folder <strong>{folderConflictModal.folderNames[0]}</strong> already exists. Uploading will overwrite any files with the same names inside it.</>
                                : <>The following folders already exist: {folderConflictModal.folderNames.map((n, i) => (
                                    <span key={n}>{i > 0 && ', '}<strong>{n}</strong></span>
                                  ))}. Uploading will overwrite any files with the same names inside them.</>}
                        </p>
                        <p style={{ margin: 0 }}>Do you want to continue?</p>
                    </div>
                )}
            </ModalWindow>

            {/* ── Bulk Move ── */}
            <ModalWindow
                isOpen={!!bulkMoveModal}
                title={`Bulk Move — ${bulkMoveModal?.items?.length ?? 0} items`}
                onOk={handleBulkMoveOk}
                onCancel={() => setBulkMoveModal(null)}
                okDisabled={!bulkMoveModal?.destPath}
                validationErrors={bulkMoveModal?.error ? [bulkMoveModal.error] : []}
            >
                {bulkMoveModal && (
                    <div>
                        <label style={labelStyle} htmlFor="bulk-move-dest">
                            Destination path within <strong>{rootFolder}</strong> (empty = root)
                        </label>
                        <input id="bulk-move-dest" style={inputStyle} autoFocus
                            placeholder="e.g. subfolder/nested"
                            value={bulkMoveModal.destPath}
                            onChange={e => setBulkMoveModal(p => ({ ...p, destPath: e.target.value, error: '' }))}
                            onKeyDown={e => e.key === 'Enter' && handleBulkMoveOk()}
                        />
                        <ul style={{ margin: '0.75rem 0 0', paddingLeft: '1.25rem', fontSize: '0.82rem', color: 'var(--blue_secondary)' }}>
                            {bulkMoveModal.items.map(i => <li key={i.path}>{i.name}</li>)}
                        </ul>
                    </div>
                )}
            </ModalWindow>

            {/* ── Bulk Delete ── */}
            <ModalWindow
                isOpen={!!bulkDeleteModal}
                title="Confirm Bulk Delete"
                onOk={handleBulkDeleteOk}
                onCancel={() => setBulkDeleteModal(null)}
                okLabel="Delete All"
                okButtonColor="var(--red_primary)"
                validationErrors={bulkDeleteModal?.error ? [bulkDeleteModal.error] : []}
            >
                {bulkDeleteModal && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <p style={{ margin: 0 }}>
                            The following <strong>{bulkDeleteModal.items.length}</strong> items will be permanently deleted:
                        </p>
                        <ul style={{
                            margin: 0,
                            paddingLeft: '1.25rem',
                            fontSize: '0.82rem',
                            color: 'var(--blue_secondary)',
                            overflowY: 'auto',
                            maxHeight: '40vh',
                            border: '1px solid #e2e8f0',
                            borderRadius: '0.375rem',
                            padding: '0.5rem 0.5rem 0.5rem 1.75rem',
                        }}>
                            {bulkDeleteModal.items.map(i => <li key={i.path}>{i.name}</li>)}
                        </ul>
                    </div>
                )}
            </ModalWindow>

            {/* ── Bulk Download ── */}
            <ModalWindow
                isOpen={!!bulkDownloadModal}
                title={`Bulk Download — ${bulkDownloadModal?.items?.length ?? 0} item${bulkDownloadModal?.items?.length === 1 ? '' : 's'}`}
                onOk={handleBulkDownloadOk}
                onCancel={() => setBulkDownloadModal(null)}
                okLabel="Download"
                validationErrors={[]}
            >
                {bulkDownloadModal && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <p style={{ margin: 0 }}>
                            The selected files and folders will be compressed into a <strong>.zip</strong> archive,
                            which may take some time depending on the total size.
                        </p>
                        <ul style={{
                            margin: 0,
                            paddingLeft: '1.25rem',
                            fontSize: '0.82rem',
                            color: 'var(--blue_secondary)',
                            overflowY: 'auto',
                            maxHeight: '40vh',
                            border: '1px solid #e2e8f0',
                            borderRadius: '0.375rem',
                            padding: '0.5rem 0.5rem 0.5rem 1.75rem',
                        }}>
                            {bulkDownloadModal.items.map(i => <li key={i.path}>{i.name}</li>)}
                        </ul>
                    </div>
                )}
            </ModalWindow>

            {/* ── Context menu ── */}
            {ctxMenu && (
                <PopupMenu
                    position={{ x: ctxMenu.x, y: ctxMenu.y }}
                    onClose={() => setCtxMenu(null)}
                    items={buildCtxMenuItems({
                        item:              ctxMenu.item,
                        canModify,
                        canCreateOrUpload,
                        isMarkdown:        ctxMenu.item.isMarkdown,
                        onPreview:         handleDoubleClick,
                        onMarkdownPreview: handleMarkdownPreview,
                        onJsxPreview:      handleJsxPreview,
                        onDownload:        handleDownload,
                        onRename:          i => setRenameModal({ item: i, newName: i.name, error: '' }),
                        onMove:            i => setMoveModal({ item: i, destPath: '', error: '' }),
                        onDelete:          i => setDeleteModal({ item: i, error: '' }),
                        onNewFolder:       () => setNewFolderModal({ name: '', error: '' }),
                        onNewFile:         () => setNewTextModal({ name: '', error: '' }),
                        onNewExternalSource: openExternalSourceModal,
                        extraForFile:      contextMenuExtraForFile,
                        extraForFolder:    contextMenuExtraForFolder,
                    })}
                    minWidth={190}
                />
            )}

        </div>
    );
};

export default FolderBrowser;
