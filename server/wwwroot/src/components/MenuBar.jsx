import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useConfig } from '../contexts/ConfigContext';
import PopupMenu from './PopupMenu';
import ModalWindow from './ModalWindow';

import mainPageIcon from '../assets/icons/mainPage.svg';
import aiAgentIcon from '../assets/icons/aiAgent.svg';
import folderIcon from '../assets/icons/folder.svg';
import schedulerIcon from '../assets/icons/scheduler.svg';
import gitIcon from '../assets/icons/git.svg';
import settingsIcon     from '../assets/icons/settings.svg';
import manageFilesIcon  from '../assets/icons/importExport.svg';
import logoutIcon       from '../assets/icons/logout.svg';
import changePasswordIcon from '../assets/icons/change_password.svg';
import auditIcon from '../assets/icons/audit.svg';
import sandboxIcon from '../assets/icons/sandbox.svg';
import fullscreenIcon from '../assets/icons/fullscreen.svg';
import fullscreenExitIcon from '../assets/icons/fullscreenExit.svg';

/** Page ids hidden when runMode is "production". */
const PRODUCTION_HIDDEN_IDS = new Set(['scheduler', 'git', 'sandbox']);

/** Page ids hidden for non-admin users. */
const ADMIN_ONLY_IDS = new Set(['settings', 'manage-files', 'audit']);

/** Full navigation tree used for current-page detection. */
export const PAGES = [
    { id: 'main', label: 'Main', path: '/main', icon: mainPageIcon },
    { id: 'ai-agent', label: 'AI Agent', path: '/', icon: aiAgentIcon },
    { id: 'data',     label: 'Data',     path: '/data', icon: folderIcon },
    { id: 'scheduler', label: 'Scheduler', path: '/scheduler', icon: schedulerIcon },
    { id: 'git',       label: 'GIT',       path: '/git',       icon: gitIcon       },
    { id: 'settings',     label: 'Settings',     path: '/settings',     icon: settingsIcon    },
    { id: 'manage-files', label: 'Manage Files', path: '/manage-files', icon: manageFilesIcon },
    { id: 'audit',     label: 'Audit',     path: '/audit',     icon: auditIcon     },
    { id: 'sandbox',   label: 'Sandbox',   path: '/sandbox',   icon: sandboxIcon   },
];

/** Flatten PAGES into leaf entries with an icon reference for easy lookup. */
function allLeafPages() {
    const result = [];
    for (const page of PAGES) {
        if (page.children) {
            for (const child of page.children) {
                result.push({ ...child, icon: page.icon });
            }
        } else {
            result.push(page);
        }
    }
    return result;
}

const LEAF_PAGES = allLeafPages();

/** Extra pages that appear in the header title but not in the navigation menu. */
const EXTRA_PAGES = [
    { id: 'change-password', label: 'Change Password', path: '/change-password', icon: changePasswordIcon },
];

/** Return the leaf page config matching the current pathname. */
function currentPage(pathname) {
    // Exact match first, then prefix (for /folders/:type)
    return (
        EXTRA_PAGES.find((p) => p.path === pathname) ||
        LEAF_PAGES.find((p) => p.path === pathname) ||
        LEAF_PAGES.find((p) => pathname.startsWith(p.path.split(':')[0])) ||
        LEAF_PAGES.find((p) => p.id === 'ai-agent') ||
        LEAF_PAGES[0]
    );
}

const BAR_HEIGHT = 56;
const CONNECTION_LOST_BAR_BG = '#550000';

const barStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: `${BAR_HEIGHT}px`,
    backgroundColor: '#1e293b',
    color: '#f1f5f9',
    padding: '0',
    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    flexShrink: 0,
};

const leftStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    overflow: 'hidden',
    padding: '0 1rem',
};

const pageNameStyle = {
    fontWeight: 600,
    fontSize: '1rem',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
};

const menuButtonStyle = {
    background: 'none',
    border: '1px solid #475569',
    color: '#f1f5f9',
    padding: '0.375rem 0.875rem',
    marginRight: '1rem',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    transition: 'background 0.15s',
};


/** Builds the items array for the navigation PopupMenu. */
function buildMenuItems(visiblePages, pathname, onNavigate, onLogout, isFullscreen, onToggleFullscreen, configLoaded) {
    const items = [];

    if (!configLoaded) {
        items.push({ label: 'Loading ...', disabled: true, color: '#e2e8f0' });
    }

    for (const p of visiblePages) {
        if (p.children) {
            items.push({
                type: 'header',
                label: p.label,
                icon: <img src={p.icon} alt="" width={20} height={20} />,
                background: '#0f172a',
                color: '#e2e8f0',
            });
            for (const child of p.children) {
                items.push({
                    label: child.label,
                    indent: true,
                    color: '#ffffff',
                    background: pathname === child.path ? '#1e3a5f' : undefined,
                    onClick: () => onNavigate(child.path),
                    allowOpenInNewTab: true,
                    href: child.path,
                });
            }
        } else {
            items.push({
                label: p.label,
                icon: <img src={p.icon} alt="" width={20} height={20} />,
                color: '#e2e8f0',
                background: pathname === p.path ? '#0f172a' : undefined,
                onClick: () => onNavigate(p.path),
                allowOpenInNewTab: true,
                href: p.path,
            });
        }

    }

    items.push({ type: 'separator' });
    items.push({
        label: 'Toggle Full-Screen',
        icon: <img src={isFullscreen ? fullscreenExitIcon : fullscreenIcon} alt="" width={20} height={20} />,
        color: '#e2e8f0',
        onClick: onToggleFullscreen,
    });
    items.push({ type: 'separator' });
    items.push({
        label: 'Change Password',
        icon: <img src={changePasswordIcon} alt="" width={20} height={20} />,
        color: '#e2e8f0',
        background: pathname === '/change-password' ? '#0f172a' : undefined,
        onClick: () => onNavigate('/change-password'),
        allowOpenInNewTab: true,
        href: '/change-password',
    });
    items.push({
        label: 'Sign Out',
        icon: <img src={logoutIcon} alt="" width={20} height={20} />,
        color: '#ffffff',
        onClick: onLogout,
    });

    return items;
}

const PING_INTERVAL_MS = 60_000;
const PING_TIMEOUT_MS  = 5_800;

/** Top navigation bar: shows current page icon + name on the left, menu dropdown on the right. */
const MenuBar = () => {
    const { logout, role } = useAuth();
    const { runMode, configLoaded } = useConfig();
    const navigate = useNavigate();
    const location = useLocation();
    const [isOpen, setIsOpen] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const menuBtnRef = useRef(null);
    const [connectionLost, setConnectionLost] = useState(false);
    const [showConnectionModal, setShowConnectionModal] = useState(false);

    useEffect(() => {
        const handleFullscreenChange = () => {
            const isNowFullscreen = !!(
                document.fullscreenElement ||
                document.webkitFullscreenElement ||
                document.mozFullScreenElement ||
                document.msFullscreenElement
            );
            setIsFullscreen(isNowFullscreen);
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('mozfullscreenchange', handleFullscreenChange);
        document.addEventListener('MSFullscreenChange', handleFullscreenChange);

        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
            document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
            document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
            document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
        };
    }, []);

    useEffect(() => {
        let cancelled = false;

        const checkConnection = async () => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), PING_TIMEOUT_MS);
            try {
                const resp = await fetch('/api/ping', { signal: controller.signal });
                if (!cancelled) setConnectionLost(!resp.ok);
            } catch {
                if (!cancelled) setConnectionLost(true);
            } finally {
                clearTimeout(timeoutId);
            }
        };

        const intervalId = setInterval(checkConnection, PING_INTERVAL_MS);

        return () => {
            cancelled = true;
            clearInterval(intervalId);
        };
    }, []);

    const toggleFullscreen = useCallback(async () => {
        try {
            if (!isFullscreen) {
                const element = document.documentElement;
                if (element.requestFullscreen) {
                    await element.requestFullscreen();
                } else if (element.webkitRequestFullscreen) {
                    await element.webkitRequestFullscreen();
                } else if (element.mozRequestFullScreen) {
                    await element.mozRequestFullScreen();
                } else if (element.msRequestFullscreen) {
                    await element.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    await document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    await document.webkitExitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    await document.mozCancelFullScreen();
                } else if (document.msExitFullscreen) {
                    await document.msExitFullscreen();
                }
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    }, [isFullscreen]);

    const isProduction = runMode === 'production';
    const isAdmin = role === 'admin';
    const visiblePages = configLoaded
        ? PAGES.filter((p) => {
            if (isProduction && PRODUCTION_HIDDEN_IDS.has(p.id)) return false;
            if (!isAdmin && ADMIN_ONLY_IDS.has(p.id)) return false;
            return true;
        })
        : [];

    const page = currentPage(location.pathname);

    const handleNavigate = (path) => navigate(path);
    const handleLogout   = () => logout();

    return (
        <>
            <header
                style={{
                    ...barStyle,
                    ...(connectionLost ? { backgroundColor: CONNECTION_LOST_BAR_BG } : {}),
                }}
            >
                {/* Left: current page icon + name */}
                <div style={leftStyle}>
                    <img src={page.icon} alt="" width={24} height={24} />
                    <span style={pageNameStyle}>{page.label}</span>
                </div>

                {/* Right: connection lost indicator + menu button */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
                    {connectionLost && (
                        <button
                            style={{
                                background: 'var(--red_primary)',
                                border: 'none',
                                color: '#ffffff',
                                padding: '0.375rem 0.875rem',
                                borderRadius: '0.375rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                fontWeight: 600,
                                whiteSpace: 'nowrap',
                            }}
                            onClick={() => setShowConnectionModal(true)}
                        >
                            Connection Lost
                        </button>
                    )}
                    <button
                        ref={menuBtnRef}
                        style={{ ...menuButtonStyle, marginRight: 0 }}
                        onClick={() => setIsOpen((v) => !v)}
                        aria-expanded={isOpen}
                        aria-haspopup="true"
                    >
                        Menu
                    </button>
                </div>

                {isOpen && (
                    <PopupMenu
                        anchor={menuBtnRef.current}
                        onClose={() => setIsOpen(false)}
                        backgroundColor="#1e293b"
                        borderColor="#334155"
                        hoverBackground="#334155"
                        minWidth={220}
                        items={buildMenuItems(visiblePages, location.pathname, handleNavigate, handleLogout, isFullscreen, toggleFullscreen, configLoaded)}
                    />
                )}
            </header>

            <ModalWindow
                isOpen={showConnectionModal}
                title="Connection Lost"
                onOk={() => setShowConnectionModal(false)}
                okLabel="OK"
            >
                <p style={{ margin: 0, lineHeight: 1.6 }}>
                    The connection to the server has been lost. The system will automatically attempt to reconnect.
                    If the problem persists, check your network connection or contact the system administrator.
                </p>
            </ModalWindow>
        </>
    );
};

export default MenuBar;
