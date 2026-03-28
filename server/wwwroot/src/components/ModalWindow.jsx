import React, { useState, useRef, useEffect } from 'react';
import Button from './Button';
import warningIcon from '../assets/icons/warning.svg';

const ModalWindow = ({
    isOpen,
    title,
    children,
    onOk,
    onCancel,
    okLabel = 'OK',
    cancelLabel = 'Cancel',
    validationErrors = [],
    validationWarnings = [],
    okDisabled = false,
    cancelDisabled = false,
    movable = true,
    headerAction = null,
    customFooterButtons = [],
    okButtonColor = null,
    resizable = false,
    style = {},
}) => {
    const [showTopShadow, setShowTopShadow] = useState(false);
    const [showBottomShadow, setShowBottomShadow] = useState(false);
    const [showWarningPopup, setShowWarningPopup] = useState(false);
    const bodyRef = useRef(null);

    // On mobile/small viewports use a smaller max height (81vh) so the modal stays on screen when scrollable
    const [modalMaxHeight, setModalMaxHeight] = useState('90vh');
    useEffect(() => {
        const mobileMaxWidth = 768;
        const updateMaxHeight = () => {
            setModalMaxHeight(window.innerWidth <= mobileMaxWidth ? '85vh' : '90vh');
        };
        updateMaxHeight();
        window.addEventListener('resize', updateMaxHeight);
        return () => window.removeEventListener('resize', updateMaxHeight);
    }, []);

    // Draggable state
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0, initialX: 0, initialY: 0 });

    // Resizable state
    const modalRef = useRef(null);
    const [modalSize, setModalSize] = useState({ width: null, height: null });
    const [isResizing, setIsResizing] = useState(false);
    const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0 });

    const checkScroll = () => {
        if (bodyRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = bodyRef.current;
            setShowTopShadow(scrollTop > 0);
            setShowBottomShadow(Math.ceil(scrollTop + clientHeight) < scrollHeight);
        }
    };

    useEffect(() => {
        if (isOpen) {
            // Check initially and on resize
            checkScroll();
            window.addEventListener('resize', checkScroll);
            return () => window.removeEventListener('resize', checkScroll);
        } else {
            // Reset position and size when closed
            setPosition({ x: 0, y: 0 });
            setModalSize({ width: null, height: null });
        }
    }, [isOpen, children]); // Re-check if children change

    // Dragging handlers
    useEffect(() => {
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            e.preventDefault();
            const dx = e.clientX - dragStartRef.current.x;
            const dy = e.clientY - dragStartRef.current.y;
            setPosition({
                x: dragStartRef.current.initialX + dx,
                y: dragStartRef.current.initialY + dy
            });
        };

        const handleMouseUp = () => {
            if (isDragging) {
                setIsDragging(false);
            }
        };

        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging]);

    const handleMouseDown = (e) => {
        if (!movable) return;
        // Only allow dragging from header, but we attach this to header div so it's implicitly true
        setIsDragging(true);
        dragStartRef.current = {
            x: e.clientX,
            y: e.clientY,
            initialX: position.x,
            initialY: position.y
        };
    };

    // Resize handlers — applying 2× delta so the corner tracks the cursor exactly
    // (the modal is flex-centered, so each 1px of growth shifts the corner only 0.5px)
    const handleResizeMouseDown = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const rect = modalRef.current.getBoundingClientRect();
        resizeStartRef.current = {
            x: e.clientX,
            y: e.clientY,
            width: rect.width,
            height: rect.height,
        };
        setIsResizing(true);
    };

    useEffect(() => {
        const handleMouseMove = (e) => {
            const dx = e.clientX - resizeStartRef.current.x;
            const dy = e.clientY - resizeStartRef.current.y;
            setModalSize({
                width: Math.max(360, resizeStartRef.current.width + 2 * dx),
                height: Math.max(280, resizeStartRef.current.height + 2 * dy),
            });
        };

        const handleMouseUp = () => setIsResizing(false);

        if (isResizing) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isResizing]);

    if (!isOpen) return null;

    const overlayStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
    };

    const modalStyle = {
        backgroundColor: '#ffffff',
        borderRadius: '0.5rem',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        width: '90%',
        maxWidth: '500px',
        display: 'flex',
        flexDirection: 'column',
        maxHeight: resizable ? 'none' : modalMaxHeight,
        position: 'relative',
        transform: `translate(${position.x}px, ${position.y}px)`,
        ...(resizable ? {
            overflow: 'hidden',
            minWidth: 360,
            minHeight: 280,
            maxWidth:  'calc(100vw - 1rem)',
            maxHeight: 'calc(100vh - 1rem)',
        } : {}),
        ...style,
        // modalSize must come after ...style so it overrides the parent's initial width/height
        ...(resizable && modalSize.width  !== null ? { width:  modalSize.width  } : {}),
        ...(resizable && modalSize.height !== null ? { height: modalSize.height } : {}),
    };

    const headerStyle = {
        padding: '1rem',
        borderBottom: '1px solid #e2e8f0',
        fontWeight: 'bold',
        fontSize: '1.25rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: showTopShadow ? '0 16px 24px -4px rgba(0, 0, 0, 0.2)' : 'none',
        zIndex: 10,
        transition: 'box-shadow 0.2s ease',
        cursor: movable ? 'move' : 'default',
        userSelect: 'none',
        flexShrink: 0,
    };

    const bodyStyle = {
        padding: '1rem',
        flex: 1,
        minHeight: 0,
        ...(resizable ? { overflow: 'hidden' } : { overflowY: 'auto' }),
    };

    const footerStyle = {
        padding: '1rem',
        borderTop: '1px solid #e2e8f0',
        display: 'flex',
        justifyContent: 'flex-end',
        gap: '1rem',
        boxShadow: showBottomShadow ? '0 -16px 24px -4px rgba(0, 0, 0, 0.2)' : 'none',
        zIndex: 10,
        transition: 'box-shadow 0.2s ease',
        flexShrink: 0,
    };

    const warningPopupStyle = {
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: '#900000',
        padding: '2rem',
        borderRadius: '0.5rem',
        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        zIndex: 20,
        width: '80%',
        maxWidth: '400px',
        border: '1px solid #fee2e2',
        color: '#ffffff'
    };

    const warningOverlayStyle = {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        zIndex: 15,
        borderRadius: '0.5rem'
    };

    const hasErrors = validationErrors.length > 0;
    const hasWarnings = validationWarnings.length > 0;
    const hasIssues = hasErrors || hasWarnings;

    return (
        <div style={overlayStyle}>
            <div style={modalStyle} ref={modalRef}>
                <div style={headerStyle} onMouseDown={handleMouseDown}>
                    <span style={{ flex: 1, minWidth: 0 }}>{title}</span>
                    {headerAction && (
                        <div
                            style={{ marginLeft: '1rem', display: 'flex', alignItems: 'center' }}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            {headerAction}
                        </div>
                    )}
                </div>

                <div
                    style={bodyStyle}
                    ref={bodyRef}
                    onScroll={checkScroll}
                >
                    {children}
                </div>

                <div style={footerStyle}>
                    {hasIssues && (
                        <Button
                            label={
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <img src={warningIcon} alt="Warning" width="20" height="20" />
                                    <span>{hasErrors ? 'Error' : 'Warning'}</span>
                                </div>
                            }
                            onClick={() => setShowWarningPopup(true)}
                            color={hasErrors ? "#900000" : "#d07000"}
                            style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                        />
                    )}
                    {onCancel && (
                        <Button
                            label={cancelLabel}
                            onClick={onCancel}
                            color={cancelDisabled ? 'var(--blue_secondary_disabled)' : 'var(--blue_secondary)'}
                            disabled={cancelDisabled}
                            style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                        />
                    )}
                    {customFooterButtons}
                    {onOk && (
                        <Button
                            label={okLabel}
                            onClick={onOk}
                            color={hasErrors || okDisabled ? 'var(--blue_primary_disabled)' : (okButtonColor ?? 'var(--blue_primary)')}
                            disabled={hasErrors || okDisabled}
                            style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                        />
                    )}
                </div>

                {resizable && (
                    <div
                        onMouseDown={handleResizeMouseDown}
                        style={{
                            position: 'absolute',
                            bottom: 0,
                            right: 0,
                            width: 18,
                            height: 18,
                            cursor: 'se-resize',
                            zIndex: 30,
                            display: 'flex',
                            alignItems: 'flex-end',
                            justifyContent: 'flex-end',
                            padding: '2px',
                            opacity: 0.35,
                            pointerEvents: 'auto',
                        }}
                    >
                        <svg width="10" height="10" viewBox="0 0 10 10" style={{ display: 'block', pointerEvents: 'none' }}>
                            <line x1="9" y1="1" x2="1" y2="9" stroke="#475569" strokeWidth="1.5" strokeLinecap="round"/>
                            <line x1="9" y1="5" x2="5" y2="9" stroke="#475569" strokeWidth="1.5" strokeLinecap="round"/>
                        </svg>
                    </div>
                )}

                {showWarningPopup && (
                    <>
                        <div style={warningOverlayStyle} onClick={() => setShowWarningPopup(false)} />
                        <div style={{
                            ...warningPopupStyle,
                            backgroundColor: hasErrors ? '#900000' : '#d07000',
                            border: hasErrors ? '1px solid #fee2e2' : '1px solid #fcd34d'
                        }}>
                            <h3 style={{ color: '#ffffff', marginTop: 0, marginBottom: '1rem' }}>
                                {hasErrors ? 'Validation Errors' : 'Warnings'}
                            </h3>
                            <ul style={{ color: '#ffffff', paddingLeft: '1.5rem', marginBottom: '1.5rem' }}>
                                {validationErrors.map((error, index) => (
                                    <li key={`err-${index}`}>{error}</li>
                                ))}
                                {validationWarnings.map((warning, index) => (
                                    <li key={`warn-${index}`}>{warning}</li>
                                ))}
                            </ul>
                            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                                <Button
                                    label="OK"
                                    onClick={() => setShowWarningPopup(false)}
                                    color="var(--blue_primary)"
                                />
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default ModalWindow;
