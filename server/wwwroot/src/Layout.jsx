import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Outlet, useLocation, Link, useNavigate } from 'react-router-dom';
import menuIcon from './assets/menu-icon.svg';
import Button from './components/Button';
import { useGeneralSettings } from './contexts/GeneralSettingsContext.jsx';
import { useAIAgentActivation } from './contexts/AIAgentActivationContext.jsx';

// Import icons
import mainPageIcon from './assets/icons/mainPage.svg';
import manualControlIcon from './assets/icons/manualControl.svg';
import aiAgentIcon from './assets/icons/aiAgent.svg';
import settingsIcon from './assets/icons/settings.svg';
import importExportIcon from './assets/icons/importExport.svg';
import motorsIcon from './assets/icons/motors.svg';
import aiBehaviorIcon from './assets/icons/aiBehavior.svg';
import hotZoneIcon from './assets/icons/hotZone.svg';
import toolsIcon from './assets/icons/tools.svg';

import cameraIcon from './assets/icons/camera.svg';
import tutorialsIcon from './assets/icons/tutorials.svg';
import systemGuideIcon from './assets/icons/systemguide.svg';
import whatToBuyIcon from './assets/icons/whattobuy.svg';
import electronicsIcon from './assets/icons/electronics.svg';
import componentsDemoIcon from './assets/icons/mainPage.svg'; // Using mainPage icon as fallback for Components Demo
import aboutIcon from './assets/icons/about.svg';
import modalWindowsIcon from './assets/icons/tip.svg';

const FULL_BLEED_PATHS = new Set([
  '/manual-control',
  '/tools/hot-zone',
  '/tutorials/system-guide',
  '/tutorials/what-to-buy',
  '/tutorials/electronics'
]);

const MenuLink = ({ to, icon, label, onNavigate }) => (
  <Link
    to={to}
    className="btn"
    style={{ justifyContent: 'flex-start', width: '100%', gap: '0.75rem' }}
    onClick={onNavigate}
  >
    {icon && <img src={icon} alt="" width="24" height="24" />}
    {label}
  </Link>
);

const Layout = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { isDebugMode } = useGeneralSettings();
  const {
    aiagentFullyActivated,
    activationLoaded,
    isActivationBusy,
    stopAIAgent
  } = useAIAgentActivation();
  const initialActivationRedirectHandledRef = useRef(false);

  /**
   * Handle menu toggle with custom event dispatch.
   * Dispatches 'menuStateChange' event for pages that need to respond (e.g., ManualControl).
   * @param {boolean} newState - The new menu state (true = open, false = closed).
   * @param {boolean} isNavigating - True if menu is closing due to navigation to another page.
   */
  const handleMenuToggle = useCallback((newState, isNavigating = false) => {
    const willOpen = typeof newState === 'boolean' ? newState : !isMenuOpen;
    setIsMenuOpen(willOpen);
    
    // Dispatch custom event for pages that need to respond to menu state
    window.dispatchEvent(new CustomEvent('menuStateChange', { 
      detail: { isOpen: willOpen, currentPath: location.pathname, isNavigating }
    }));
  }, [isMenuOpen, location.pathname]);

  useEffect(() => {
    if (!activationLoaded || initialActivationRedirectHandledRef.current) {
      return;
    }

    initialActivationRedirectHandledRef.current = true;
    if (aiagentFullyActivated && location.pathname !== '/ai-agent') {
      navigate('/ai-agent', { replace: true });
    }
  }, [activationLoaded, aiagentFullyActivated, location.pathname, navigate]);

  const handleStopAIClick = useCallback(async () => {
    handleMenuToggle(false);

    try {
      await stopAIAgent();
      navigate('/ai-agent');
    } catch (error) {
      console.error('Failed to stop AI agent:', error);
    }
  }, [handleMenuToggle, navigate, stopAIAgent]);

  const getPageInfo = () => {
    const path = location.pathname;
    if (path === '/') return { title: 'Main Page', icon: mainPageIcon };
    if (path === '/manual-control') return { title: 'Manual Control', icon: manualControlIcon };
    if (path === '/ai-agent') return { title: 'AI Agent', icon: aiAgentIcon };
    if (path.startsWith('/settings')) {
      if (path === '/settings/import-export') return { title: 'Import / Export', icon: importExportIcon };
      if (path === '/settings/general-setup') return { title: 'General Setup', icon: settingsIcon };
      if (path === '/settings/motors') return { title: 'Motors', icon: motorsIcon };
      if (path === '/settings/cameras-new') return { title: 'Cameras', icon: cameraIcon };
      if (path === '/settings/ai-behavior') return { title: 'AI Setup', icon: aiBehaviorIcon };
      return { title: 'Settings', icon: settingsIcon };
    }
    if (path.startsWith('/tools')) {
      if (path === '/tools/hot-zone') return { title: 'Hot Zone', icon: hotZoneIcon };
      return { title: 'Tools', icon: toolsIcon };
    }
    if (path.startsWith('/sandbox')) {
      if (path === '/sandbox/components-demo') return { title: 'Components Demo', icon: componentsDemoIcon };
      if (path === '/sandbox/modal-windows-demo') return { title: 'Modal Windows Demo', icon: modalWindowsIcon };
      if (path === '/sandbox/editable-chart') return { title: 'Editable Chart', icon: componentsDemoIcon };
      if (path === '/sandbox/table-demo') return { title: 'Table Demo', icon: componentsDemoIcon };
      if (path === '/sandbox/chart2d') return { title: 'Chart2D Demo', icon: componentsDemoIcon };
      if (path === '/sandbox/polygon-zoom-pan') return { title: 'Polygon Zoom Pan', icon: componentsDemoIcon };
      if (path === '/sandbox/datetimepicker') return { title: 'DateTimePicker Demo', icon: componentsDemoIcon };
      if (path === '/sandbox/joystick1d') return { title: 'Joystick1D Demo', icon: componentsDemoIcon };
      return { title: 'Developers Sandbox', icon: manualControlIcon };
    }
    if (path.startsWith('/tutorials')) {
      if (path === '/tutorials/system-guide') return { title: 'System Guide', icon: systemGuideIcon };
      if (path === '/tutorials/what-to-buy') return { title: 'What to Buy', icon: whatToBuyIcon };
      if (path === '/tutorials/electronics') return { title: 'Electronics', icon: electronicsIcon };
      return { title: 'Tutorials', icon: tutorialsIcon };
    }
    if (path === '/about') return { title: 'About', icon: aboutIcon };
    return { title: 'Submoamoa', icon: null };
  };

  const { title, icon } = getPageInfo();

  // Pages that should use full-bleed layout (no padding, no scrollbars)
  const isAIAgentFullBleed = location.pathname === '/ai-agent' && (!activationLoaded || aiagentFullyActivated);
  const isFullBleedPage = FULL_BLEED_PATHS.has(location.pathname) || isAIAgentFullBleed;
  const handleMenuNavigate = () => handleMenuToggle(false, true);

  return (
    <div className="app-container">
      <header className="app-header glass">
        <div className="header-left">
          {icon && <img src={icon} alt="" width="24" height="24" />}
          <span>{title}</span>
        </div>
        <div className="header-right">
          {aiagentFullyActivated ? (
            <Button
              label={(
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                  <img src={aiAgentIcon} alt="" width="24" height="24" />
                  <span>Stop AI</span>
                </span>
              )}
              onClick={handleStopAIClick}
              disabled={isActivationBusy}
              color="#dc2626"
              hint="Stop AI"
              style={{ width: 'auto', padding: '0.5rem 0.75rem' }}
            />
          ) : (
            <button
              className="btn menu-btn"
              onClick={() => handleMenuToggle()}
              aria-label="Menu"
            >
              <span style={{ marginRight: '0.5rem', display: 'none' }} className="menu-text">Menu</span>
              <img src={menuIcon} alt="Menu" width="24" height="24" />
            </button>
          )}
        </div>
      </header>

      {!aiagentFullyActivated && isMenuOpen && (
        <>
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              zIndex: 48
            }}
            onClick={() => handleMenuToggle(false)}
          />
          <nav className="mobile-menu glass" style={{
            position: 'fixed',
            top: 'calc(var(--header-height) + var(--safe-area-top, 0px))',
            right: 0,
            width: '250px', // Increased width for icons
            padding: '1rem',
            paddingRight: 'calc(1rem + var(--safe-area-right, 0px))',
            borderLeft: '1px solid var(--color-border)',
            borderBottom: '1px solid var(--color-border)',
            zIndex: 49,
            //maxHeight: 'calc(100vh - var(--header-height) - var(--safe-area-top, 0px))',
            maxHeight: 'calc(100dvh - var(--header-height) - var(--safe-area-top, 0px))',
            overflowY: 'auto'
          }}>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li><MenuLink to="/" icon={mainPageIcon} label="Main Page" onNavigate={handleMenuNavigate} /></li>
              <li><MenuLink to="/manual-control" icon={manualControlIcon} label="Manual Control" onNavigate={handleMenuNavigate} /></li>
              <li><MenuLink to="/ai-agent" icon={aiAgentIcon} label="AI Agent" onNavigate={handleMenuNavigate} /></li>
              <li>
                <div className="btn" style={{ justifyContent: 'flex-start', width: '100%', cursor: 'default', opacity: 0.8, gap: '0.75rem' }}>
                  <img src={settingsIcon} alt="" width="24" height="24" />
                  Settings
                </div>
                <ul style={{ listStyle: 'none', paddingLeft: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <li><MenuLink to="/settings/import-export" icon={importExportIcon} label="Import / Export" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/settings/general-setup" icon={settingsIcon} label="General Setup" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/settings/motors" icon={motorsIcon} label="Motors" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/settings/cameras-new" icon={cameraIcon} label="Cameras" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/settings/ai-behavior" icon={aiBehaviorIcon} label="AI Setup" onNavigate={handleMenuNavigate} /></li>
                </ul>
              </li>
              <li>
                <div className="btn" style={{ justifyContent: 'flex-start', width: '100%', cursor: 'default', opacity: 0.8, gap: '0.75rem' }}>
                  <img src={toolsIcon} alt="" width="24" height="24" />
                  Tools
                </div>
                <ul style={{ listStyle: 'none', paddingLeft: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <li><MenuLink to="/tools/hot-zone" icon={hotZoneIcon} label="Hot Zone" onNavigate={handleMenuNavigate} /></li>
                </ul>
              </li>
              <li>
                <div className="btn" style={{ justifyContent: 'flex-start', width: '100%', cursor: 'default', opacity: 0.8, gap: '0.75rem' }}>
                  <img src={tutorialsIcon} alt="" width="24" height="24" />
                  Tutorials
                </div>
                <ul style={{ listStyle: 'none', paddingLeft: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <li><MenuLink to="/tutorials/system-guide" icon={systemGuideIcon} label="System Guide" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/tutorials/what-to-buy" icon={whatToBuyIcon} label="What to Buy" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/tutorials/electronics" icon={electronicsIcon} label="Electronics" onNavigate={handleMenuNavigate} /></li>
                </ul>
              </li>
              {isDebugMode && (
              <li>
                <div className="btn" style={{ justifyContent: 'flex-start', width: '100%', cursor: 'default', opacity: 0.8, gap: '0.75rem' }}>
                  <img src={manualControlIcon} alt="" width="24" height="24" />
                  Developers Sandbox
                </div>
                <ul style={{ listStyle: 'none', paddingLeft: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <li><MenuLink to="/sandbox/components-demo" icon={componentsDemoIcon} label="Components Demo" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/modal-windows-demo" icon={modalWindowsIcon} label="Modal Windows Demo" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/editable-chart" icon={componentsDemoIcon} label="Editable Chart" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/table-demo" icon={componentsDemoIcon} label="Table Demo" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/chart2d" icon={componentsDemoIcon} label="Chart2D Demo" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/polygon-zoom-pan" icon={componentsDemoIcon} label="Polygon Zoom Pan" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/datetimepicker" icon={componentsDemoIcon} label="DateTimePicker Demo" onNavigate={handleMenuNavigate} /></li>
                  <li><MenuLink to="/sandbox/joystick1d" icon={componentsDemoIcon} label="Joystick1D Demo" onNavigate={handleMenuNavigate} /></li>
                </ul>
              </li>
              )}
              <li><MenuLink to="/about" icon={aboutIcon} label="About" onNavigate={handleMenuNavigate} /></li>
            </ul>
          </nav>
        </>
      )}

      <main className={`main-content${isFullBleedPage ? ' full-bleed' : ''}`}>
        <Outlet />
      </main>

      <style>{`
        @media (min-width: 640px) {
          .menu-text {
            display: inline !important;
          }
        }
      `}</style>
    </div>
  );
};

export default Layout;
