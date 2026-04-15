import React from 'react';
import Polygon from './Polygon';
import fullscreenIcon from '../assets/icons/fullscreen.svg';
import fullscreenExitIcon from '../assets/icons/fullscreenExit.svg';
import cameraOffIcon from '../assets/icons/cameraOff.svg';

const fullscreenButtonStyle = {
    position: 'absolute',
    bottom: '16px',
    zIndex: 10,
    width: '48px',
    height: '48px',
    padding: '8px',
    backgroundColor: '#887700',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.8,
    transition: 'opacity 0.2s, background-color 0.2s'
};

function LoadingViewport() {
    return (
        <div style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#000'
        }}>
            <span style={{ color: '#fff' }}>Loading...</span>
        </div>
    );
}

const CameraViewport = ({
    isLoading = false,
    streamUrl,
    reticleSettings,
    polygonMode = 'viewer',
    zoomPanEnabled = true,
    showReticle = true,
    isFullscreen = false,
    onToggleFullscreen,
    fullscreenButtonRight = '16px',
    onJoystickMove,
    onJoystickStart,
    onJoystickEnd,
    children
}) => {
    if (isLoading) {
        return <LoadingViewport />;
    }

    return (
        <div style={{
            width: '100%',
            height: '100%',
            margin: 0,
            padding: 0,
            overflow: 'hidden',
            position: 'relative'
        }}>
            <Polygon
                src={streamUrl || cameraOffIcon}
                stretchMode="fit"
                background="#000000"
                mode={polygonMode}
                zoomPanEnabled={zoomPanEnabled}
                showReticle={showReticle}
                reticleX={reticleSettings?.x ?? 0.5}
                reticleY={reticleSettings?.y ?? 0.5}
                reticleColor={reticleSettings?.color ?? '#88ff00cc'}
                reticleOutlineColor={reticleSettings?.outline ?? '#000000cc'}
                reticleSize={reticleSettings?.size ?? 1.0}
                joystickLineMaxLength={0.33}
                onJoystickMove={onJoystickMove}
                onJoystickStart={onJoystickStart}
                onJoystickEnd={onJoystickEnd}
                style={{
                    width: '100%',
                    height: '100%'
                }}
            />

            {children}

            {typeof onToggleFullscreen === 'function' && (
                <button
                    onClick={onToggleFullscreen}
                    style={{
                        ...fullscreenButtonStyle,
                        right: fullscreenButtonRight
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.opacity = '1';
                        e.currentTarget.style.backgroundColor = '#885500';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.opacity = '0.8';
                        e.currentTarget.style.backgroundColor = '#887700';
                    }}
                    title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                    <img
                        src={isFullscreen ? fullscreenExitIcon : fullscreenIcon}
                        alt={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
                        width="24"
                        height="24"
                    />
                </button>
            )}
        </div>
    );
};

export default CameraViewport;
