import React, { useState, useRef, useEffect, useCallback } from 'react';

// Stable empty array reference to prevent infinite re-renders when no polygons prop is passed
const EMPTY_POLYGONS = [];

/**
 * Polygon component for drawing and editing polygons over an image.
 * 
 * Interaction:
 * - Click to add points when drawing
 * - Click on first point to close the polygon
 * - After closing: drag points to move them
 * - Double-click on polygon to delete it and start over
 */
const Polygon = ({
    border = 0,
    src,
    stretchMode = 'fit',
    background = 'transparent',
    style = {},
    borderColor = '#009900ff',
    fillColor = '#00ee0055',
    lineWidth = 1,
    maxPoints = 32,
    maxPolygons = 8,
    polygons: externalPolygons = EMPTY_POLYGONS,
    onChange,
    // Reticle props
    reticleX = 0.5,
    reticleY = 0.5,
    showReticle = false,
    reticleSize = 1,
    reticleColor = '#88ff00cc',
    reticleOutlineColor = '#009900cc',
    // Mode: 'viewer' (view only), 'designer' (draw polygons), 'joystick'
    mode = 'designer',
    // Joystick props
    joystickColor = '#999999cc',
    joystickSize = 24, // ~0.5 inches at 96 DPI
    joystickLineWidth = 4,
    joystickLineMaxLength = 0.25, // 25% of smaller dimension
    joystickLineMaxLengthMultiplierMode = 'minWidthHeight', // 'minWidthHeight' | 'width' | 'height'
    joystickSnapAnimationDuration = 0.1, // seconds
    joystickLineColor1 = '#555555ff', // color when dynamic on static
    joystickLineColor2 = '#ff0000ff', // color when at max distance
    joystickZeroRadius = 0.001, // dead zone radius (fraction of max length)
    joystickMoveInterval = 0, // interval in ms to throttle onJoystickMove calls (0 = no throttle)
    onJoystickMove, // callback with { x, y } normalized offset from center
    onJoystickStart, // callback when joystick becomes active
    onJoystickEnd, // callback when joystick is released
    // Zoom and pan props
    zoomPanEnabled = false
}) => {
    const [polygons, setPolygons] = useState(externalPolygons);
    const [currentPolygon, setCurrentPolygon] = useState(null); // Points being drawn (not yet closed)
    const [draggingPoint, setDraggingPoint] = useState(null); // { polyIndex, pointIndex }

    // Joystick state
    const [joystickStatic, setJoystickStatic] = useState(null); // { x, y } normalized
    const [joystickDynamic, setJoystickDynamic] = useState(null); // { x, y } normalized
    const [isJoystickActive, setIsJoystickActive] = useState(false);
    const joystickAnimationRef = useRef(null);

    // Joystick move throttling state
    const lastJoystickMoveTimeRef = useRef(0);
    const pendingJoystickMoveRef = useRef(null); // stores pending {x, y} to send
    const joystickThrottleTimeoutRef = useRef(null);
    const lastSentJoystickCoordsRef = useRef(null); // stores last sent {x, y}
    const joystickReleasedRef = useRef(false); // prevents throttle callbacks after release
    const joystickHadMovementRef = useRef(false); // tracks if there was movement since start

    // Zoom state
    const [zoom, setZoom] = useState(1);
    const [zoomCenter, setZoomCenter] = useState({ x: 0.5, y: 0.5 }); // Normalized center point for zoom
    const pinchStartDistanceRef = useRef(null);
    const pinchStartZoomRef = useRef(1);
    const pinchCenterRef = useRef({ x: 0, y: 0 });
    const pinchStartZoomCenterRef = useRef({ x: 0.5, y: 0.5 }); // Store initial zoomCenter at pinch start

    // Middle-button pan state
    const [isPanning, setIsPanning] = useState(false);
    const panStartRef = useRef({ x: 0, y: 0 }); // Mouse position at pan start
    const panStartZoomCenterRef = useRef({ x: 0.5, y: 0.5 }); // Store initial zoomCenter at pan start

    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const imageRef = useRef(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

    // Sync with external polygons
    useEffect(() => {
        setPolygons(externalPolygons);
    }, [externalPolygons]);

    // Handle image load
    useEffect(() => {
        if (imageRef.current && imageRef.current.complete) {
            handleImageLoad();
        }
    }, [src]);

    const handleImageLoad = () => {
        if (imageRef.current) {
            setImageSize({
                width: imageRef.current.naturalWidth,
                height: imageRef.current.naturalHeight
            });
            setImageLoaded(true);
        }
    };

    // Update container size on resize
    useEffect(() => {
        let animationFrameId;

        const updateSize = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setContainerSize(prev => {
                    if (Math.abs(prev.width - rect.width) < 1 && Math.abs(prev.height - rect.height) < 1) {
                        return prev;
                    }
                    return { width: rect.width, height: rect.height };
                });
            }
        };

        const onResize = () => {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = requestAnimationFrame(updateSize);
        };

        updateSize();
        window.addEventListener('resize', onResize);

        return () => {
            window.removeEventListener('resize', onResize);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    // Get clientX/Y from either mouse or touch event
    const getClientCoords = useCallback((e) => {
        if (e.touches && e.touches.length > 0) {
            return { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY };
        }
        if (e.changedTouches && e.changedTouches.length > 0) {
            return { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY };
        }
        return { clientX: e.clientX, clientY: e.clientY };
    }, []);

    // Convert screen coordinates to normalized coordinates (0-1 range relative to container)
    // Accounts for zoom and pan
    const screenToNormalized = useCallback((screenX, screenY) => {
        if (!containerRef.current) return { x: 0, y: 0 };
        const rect = containerRef.current.getBoundingClientRect();
        
        // Base normalized coords relative to viewport
        let x = (screenX - rect.left) / rect.width;
        let y = (screenY - rect.top) / rect.height;

        // Apply inverse zoom transform if enabled and zoomed
        if (zoomPanEnabled && zoom !== 1) {
            // Transform: screen = center + (image - center) * zoom
            // Inverse: image = center + (screen - center) / zoom
            x = zoomCenter.x + (x - zoomCenter.x) / zoom;
            y = zoomCenter.y + (y - zoomCenter.y) / zoom;
        }

        return {
            x: Math.max(0, Math.min(1, x)),
            y: Math.max(0, Math.min(1, y))
        };
    }, [zoomPanEnabled, zoom, zoomCenter]);

    // Convert normalized coordinates to canvas coordinates
    // Apply zoom and pan transform to convert image coordinates to screen/canvas coordinates
    const normalizedToCanvas = useCallback((normX, normY) => {
        if (!canvasRef.current) return { x: 0, y: 0 };
        
        let screenNormX = normX;
        let screenNormY = normY;

        // Apply forward zoom transform if enabled and zoomed
        if (zoomPanEnabled && zoom !== 1) {
            // Transform: screen = center + (image - center) * zoom
            screenNormX = zoomCenter.x + (normX - zoomCenter.x) * zoom;
            screenNormY = zoomCenter.y + (normY - zoomCenter.y) * zoom;
        }

        return {
            x: screenNormX * canvasRef.current.width,
            y: screenNormY * canvasRef.current.height
        };
    }, [zoomPanEnabled, zoom, zoomCenter]);

    // Convert normalized coordinates to screen coordinates
    // Accounts for zoom and pan
    const normalizedToScreen = useCallback((normX, normY) => {
        if (!containerRef.current) return { x: 0, y: 0 };
        const rect = containerRef.current.getBoundingClientRect();
        
        let screenNormX = normX;
        let screenNormY = normY;

        // Apply forward zoom transform if enabled and zoomed
        if (zoomPanEnabled && zoom !== 1) {
            // Transform: screen = center + (image - center) * zoom
            screenNormX = zoomCenter.x + (normX - zoomCenter.x) * zoom;
            screenNormY = zoomCenter.y + (normY - zoomCenter.y) * zoom;
        }

        return {
            x: screenNormX * rect.width + rect.left,
            y: screenNormY * rect.height + rect.top
        };
    }, [zoomPanEnabled, zoom, zoomCenter]);

    // Get reference dimension for joystick max length calculation based on mode
    const getJoystickReferenceDimension = useCallback(() => {
        switch (joystickLineMaxLengthMultiplierMode) {
            case 'width':
                return containerSize.width;
            case 'height':
                return containerSize.height;
            case 'minWidthHeight':
            default:
                return Math.min(containerSize.width, containerSize.height);
        }
    }, [joystickLineMaxLengthMultiplierMode, containerSize]);

    // Get content dimensions based on stretch mode (for zoom/pan calculations)
    const getContentDimensions = useCallback(() => {
        if (!containerSize.width || !containerSize.height) {
            return { width: 0, height: 0 };
        }

        if (stretchMode === 'stretch') {
            // In stretch mode, content fills the container
            return { width: containerSize.width, height: containerSize.height };
        }

        // For 'fit' and 'originalSize', use image aspect ratio
        if (!imageSize.width || !imageSize.height) {
            return { width: containerSize.width, height: containerSize.height };
        }

        const containerAspect = containerSize.width / containerSize.height;
        const imageAspect = imageSize.width / imageSize.height;

        if (stretchMode === 'originalSize') {
            // Original size, but capped at container
            const width = Math.min(imageSize.width, containerSize.width);
            const height = Math.min(imageSize.height, containerSize.height);
            return { width, height };
        }

        // 'fit' mode - maintain aspect ratio
        if (imageAspect > containerAspect) {
            // Image is wider than container
            return { width: containerSize.width, height: containerSize.width / imageAspect };
        } else {
            // Image is taller than container
            return { width: containerSize.height * imageAspect, height: containerSize.height };
        }
    }, [containerSize, imageSize, stretchMode]);

    // Handle zoom change with center point (in screen coordinates)
    const handleZoomChange = useCallback((newZoom, centerX = null, centerY = null) => {
        const clampedZoom = Math.max(1, Math.min(10, newZoom)); // Min 1x, max 10x

        if (centerX !== null && centerY !== null && containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();

            // Calculate cursor position as normalized (0-1) container coordinates
            const cursorNormX = (centerX - rect.left) / rect.width;
            const cursorNormY = (centerY - rect.top) / rect.height;

            // Calculate where the cursor points in "content space" given current zoom and zoomCenter
            // The transform is: scale(zoom) with origin at zoomCenter
            // A point in content space maps to screen as: screenPos = zoomCenter + (contentPos - zoomCenter) * zoom
            // So contentPos = zoomCenter + (screenPos - zoomCenter) / zoom
            const contentPointX = zoomCenter.x + (cursorNormX - zoomCenter.x) / zoom;
            const contentPointY = zoomCenter.y + (cursorNormY - zoomCenter.y) / zoom;

            // Now we want to find the new zoomCenter such that contentPoint maps to cursorNorm at newZoom
            // cursorNormX = newZoomCenter.x + (contentPointX - newZoomCenter.x) * newZoom
            // Solving for newZoomCenter.x:
            // cursorNormX = newZoomCenter.x + contentPointX * newZoom - newZoomCenter.x * newZoom
            // cursorNormX = newZoomCenter.x * (1 - newZoom) + contentPointX * newZoom
            // newZoomCenter.x = (cursorNormX - contentPointX * newZoom) / (1 - newZoom)
            // But this has a singularity at newZoom = 1, so let's rearrange:
            // newZoomCenter.x * (1 - newZoom) = cursorNormX - contentPointX * newZoom
            // newZoomCenter.x = cursorNormX + (cursorNormX - contentPointX * newZoom) / (newZoom - 1)  -- still problematic

            // Alternative approach: keep the content point under cursor stable
            // For zoom from oldZoom to newZoom, if contentPoint should stay at cursorNorm:
            // cursorNormX = newZoomCenterX + (contentPointX - newZoomCenterX) * clampedZoom
            // cursorNormX = newZoomCenterX * (1 - clampedZoom) + contentPointX * clampedZoom
            // newZoomCenterX * (1 - clampedZoom) = cursorNormX - contentPointX * clampedZoom
            // newZoomCenterX = (cursorNormX - contentPointX * clampedZoom) / (1 - clampedZoom)

            let newZoomCenterX, newZoomCenterY;

            if (clampedZoom === 1) {
                // At zoom 1, center doesn't matter, reset to middle
                newZoomCenterX = 0.5;
                newZoomCenterY = 0.5;
            } else {
                newZoomCenterX = (cursorNormX - contentPointX * clampedZoom) / (1 - clampedZoom);
                newZoomCenterY = (cursorNormY - contentPointY * clampedZoom) / (1 - clampedZoom);

                // Clamp zoom center to valid range [0, 1] to allow panning to image edges
                // At zoomCenter=0, the left edge of the image aligns with the left of the view
                // At zoomCenter=1, the right edge of the image aligns with the right of the view
                newZoomCenterX = Math.max(0, Math.min(1, newZoomCenterX));
                newZoomCenterY = Math.max(0, Math.min(1, newZoomCenterY));
            }

            setZoomCenter({ x: newZoomCenterX, y: newZoomCenterY });
        }

        setZoom(clampedZoom);
    }, [zoom, zoomCenter]);

    // Reset zoom
    const resetZoom = useCallback(() => {
        setZoom(1);
        setZoomCenter({ x: 0.5, y: 0.5 });
    }, []);

    // Check if screen point is near a normalized point
    const isNearPoint = (screenX, screenY, normPoint, threshold = 10) => {
        const screenPoint = normalizedToScreen(normPoint.x, normPoint.y);
        const dx = screenX - screenPoint.x;
        const dy = screenY - screenPoint.y;
        return Math.sqrt(dx * dx + dy * dy) < threshold;
    };

    // Check if point is inside polygon (using normalized coordinates)
    const isPointInPolygon = (normPoint, polygon) => {
        if (polygon.length < 3) return false;
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            const intersect = ((yi > normPoint.y) !== (yj > normPoint.y)) &&
                (normPoint.x < (xj - xi) * (normPoint.y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    };

    // Find point at screen coordinates
    const findPointAt = (screenX, screenY) => {
        // Check current polygon being drawn
        if (currentPolygon) {
            for (let i = 0; i < currentPolygon.length; i++) {
                if (isNearPoint(screenX, screenY, currentPolygon[i])) {
                    return { polyIndex: -1, pointIndex: i }; // -1 indicates current polygon
                }
            }
        }
        // Check closed polygons
        for (let polyIndex = 0; polyIndex < polygons.length; polyIndex++) {
            const polygon = polygons[polyIndex];
            for (let pointIndex = 0; pointIndex < polygon.length; pointIndex++) {
                if (isNearPoint(screenX, screenY, polygon[pointIndex])) {
                    return { polyIndex, pointIndex };
                }
            }
        }
        return null;
    };

    // Find polygon at screen coordinates
    const findPolygonAt = (screenX, screenY) => {
        const normCoords = screenToNormalized(screenX, screenY);
        for (let i = polygons.length - 1; i >= 0; i--) {
            if (isPointInPolygon(normCoords, polygons[i])) {
                return i;
            }
        }
        return null;
    };

    // Draw everything on canvas
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        if (src && !imageLoaded) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw closed polygons
        // Point size and line thickness remain constant in screen pixels (not scaled with zoom)
        polygons.forEach((polygon) => {
            if (polygon.length < 3) return;

            ctx.beginPath();
            const first = normalizedToCanvas(polygon[0].x, polygon[0].y);
            ctx.moveTo(first.x, first.y);

            for (let i = 1; i < polygon.length; i++) {
                const pt = normalizedToCanvas(polygon[i].x, polygon[i].y);
                ctx.lineTo(pt.x, pt.y);
            }
            ctx.closePath();

            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.strokeStyle = borderColor;
            // Line width remains constant in screen pixels
            ctx.lineWidth = lineWidth;
            ctx.stroke();

            // Draw vertex points (constant size in screen pixels)
            polygon.forEach((point) => {
                const canvasPt = normalizedToCanvas(point.x, point.y);
                ctx.beginPath();
                ctx.arc(canvasPt.x, canvasPt.y, 5, 0, Math.PI * 2);
                ctx.fillStyle = borderColor;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        });

        // Draw current polygon being drawn
        // Point size and line thickness remain constant in screen pixels (not scaled with zoom)
        if (currentPolygon && currentPolygon.length > 0) {
            ctx.beginPath();
            const first = normalizedToCanvas(currentPolygon[0].x, currentPolygon[0].y);
            ctx.moveTo(first.x, first.y);

            for (let i = 1; i < currentPolygon.length; i++) {
                const pt = normalizedToCanvas(currentPolygon[i].x, currentPolygon[i].y);
                ctx.lineTo(pt.x, pt.y);
            }

            ctx.strokeStyle = borderColor;
            // Line width remains constant in screen pixels
            ctx.lineWidth = lineWidth;
            ctx.stroke();

            // Draw vertex points (constant size in screen pixels)
            currentPolygon.forEach((point, index) => {
                const canvasPt = normalizedToCanvas(point.x, point.y);
                ctx.beginPath();
                ctx.arc(canvasPt.x, canvasPt.y, index === 0 ? 8 : 5, 0, Math.PI * 2);
                ctx.fillStyle = index === 0 ? '#ff6600' : borderColor; // First point highlighted
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }

        // Draw reticle if enabled
        // Reticle size remains constant in screen pixels (not scaled with zoom)
        if (showReticle) {
            const centerPos = normalizedToCanvas(reticleX, reticleY);
            
            // Reticle dimensions are constant in screen pixels
            const baseSize = 20 * reticleSize;
            const gapSize = 12 * reticleSize;
            const lineLength = 14 * reticleSize;
            const dotRadius = 2 * reticleSize;
            const reticleLineWidth = 2 * reticleSize;
            const outlineWidth = Math.max(0.5, 0.5 * reticleLineWidth);

            ctx.save();
            ctx.lineCap = 'round';

            // Draw outline for cross lines first (behind main lines)
            // Outline stroke = reticleLineWidth + 2 * outlineWidth (outlineWidth on each side)
            ctx.strokeStyle = reticleOutlineColor;
            ctx.lineWidth = reticleLineWidth + 2 * outlineWidth;

            // Top line outline
            ctx.beginPath();
            ctx.moveTo(centerPos.x, centerPos.y - gapSize);
            ctx.lineTo(centerPos.x, centerPos.y - gapSize - lineLength);
            ctx.stroke();

            // Bottom line outline
            ctx.beginPath();
            ctx.moveTo(centerPos.x, centerPos.y + gapSize);
            ctx.lineTo(centerPos.x, centerPos.y + gapSize + lineLength);
            ctx.stroke();

            // Left line outline
            ctx.beginPath();
            ctx.moveTo(centerPos.x - gapSize, centerPos.y);
            ctx.lineTo(centerPos.x - gapSize - lineLength, centerPos.y);
            ctx.stroke();

            // Right line outline
            ctx.beginPath();
            ctx.moveTo(centerPos.x + gapSize, centerPos.y);
            ctx.lineTo(centerPos.x + gapSize + lineLength, centerPos.y);
            ctx.stroke();

            // Draw main cross lines on top
            ctx.strokeStyle = reticleColor;
            ctx.fillStyle = reticleColor;
            ctx.lineWidth = reticleLineWidth;

            // Top line
            ctx.beginPath();
            ctx.moveTo(centerPos.x, centerPos.y - gapSize);
            ctx.lineTo(centerPos.x, centerPos.y - gapSize - lineLength);
            ctx.stroke();

            // Bottom line
            ctx.beginPath();
            ctx.moveTo(centerPos.x, centerPos.y + gapSize);
            ctx.lineTo(centerPos.x, centerPos.y + gapSize + lineLength);
            ctx.stroke();

            // Left line
            ctx.beginPath();
            ctx.moveTo(centerPos.x - gapSize, centerPos.y);
            ctx.lineTo(centerPos.x - gapSize - lineLength, centerPos.y);
            ctx.stroke();

            // Right line
            ctx.beginPath();
            ctx.moveTo(centerPos.x + gapSize, centerPos.y);
            ctx.lineTo(centerPos.x + gapSize + lineLength, centerPos.y);
            ctx.stroke();

            // Draw center dot (no outline)
            ctx.beginPath();
            ctx.arc(centerPos.x, centerPos.y, dotRadius, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();
        }

        // Draw joystick if active
        // Joystick size is constant in screen pixels (not scaled with zoom)
        if (mode === 'joystick' && joystickStatic && joystickDynamic) {
            const staticPos = normalizedToCanvas(joystickStatic.x, joystickStatic.y);
            const dynamicPos = normalizedToCanvas(joystickDynamic.x, joystickDynamic.y);
            const radius = joystickSize / 2;

            ctx.save();

            // Calculate distance ratio for color interpolation
            const dx = dynamicPos.x - staticPos.x;
            const dy = dynamicPos.y - staticPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const referenceDim = getJoystickReferenceDimension();
            // Max distance is constant in screen pixels
            const maxDist = referenceDim * joystickLineMaxLength;
            const distRatio = Math.min(1, dist / maxDist);

            // Interpolate line color
            const parseHexColor = (hex) => {
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                const a = hex.length >= 9 ? parseInt(hex.slice(7, 9), 16) / 255 : 1;
                return { r, g, b, a };
            };
            const color1 = parseHexColor(joystickLineColor1);
            const color2 = parseHexColor(joystickLineColor2);
            const interpColor = {
                r: Math.round(color1.r + (color2.r - color1.r) * distRatio),
                g: Math.round(color1.g + (color2.g - color1.g) * distRatio),
                b: Math.round(color1.b + (color2.b - color1.b) * distRatio),
                a: color1.a + (color2.a - color1.a) * distRatio
            };
            const lineColor = `rgba(${interpColor.r}, ${interpColor.g}, ${interpColor.b}, ${interpColor.a})`;

            // Draw dotted line connecting centers
            ctx.beginPath();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = joystickLineWidth;
            ctx.moveTo(staticPos.x, staticPos.y);
            ctx.lineTo(dynamicPos.x, dynamicPos.y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw static circle (bottom)
            ctx.beginPath();
            ctx.arc(staticPos.x, staticPos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = joystickColor;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw dynamic circle (top)
            ctx.beginPath();
            ctx.arc(dynamicPos.x, dynamicPos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = joystickColor;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
        }
    }, [polygons, currentPolygon, imageLoaded, normalizedToCanvas, borderColor, fillColor, lineWidth, showReticle, reticleX, reticleY, reticleSize, reticleColor, reticleOutlineColor, mode, joystickStatic, joystickDynamic, joystickColor, joystickSize, joystickLineWidth, joystickLineColor1, joystickLineColor2, joystickLineMaxLength, containerSize, getJoystickReferenceDimension, zoom, zoomPanEnabled]);

    useEffect(() => {
        draw();
    }, [draw]);

    // Cleanup timeouts and animations on unmount
    useEffect(() => {
        return () => {
            if (joystickThrottleTimeoutRef.current) {
                clearTimeout(joystickThrottleTimeoutRef.current);
            }
            if (joystickAnimationRef.current) {
                cancelAnimationFrame(joystickAnimationRef.current);
            }
        };
    }, []);

    // Ref to always have the latest wheel handler (avoids stale closure in native event listener)
    const wheelHandlerRef = useRef(null);

    // Handle wheel zoom
    const handleWheel = useCallback((e) => {
        if (!zoomPanEnabled) return;

        // Prevent page/modal scrolling and stop event propagation
        e.preventDefault();
        e.stopPropagation();
        
        const zoomFactor = 0.1;
        const delta = e.deltaY > 0 ? -zoomFactor : zoomFactor;
        const newZoom = zoom * (1 + delta);

        handleZoomChange(newZoom, e.clientX, e.clientY);
    }, [zoomPanEnabled, zoom, handleZoomChange]);

    // Keep the ref updated with the latest handler
    wheelHandlerRef.current = handleWheel;

    // Handle touch events for pinch-to-zoom
    const handlePinchTouchStart = useCallback((e) => {
        if (!zoomPanEnabled) return;
        if (e.touches.length !== 2) return;

        // Calculate initial distance between two fingers
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const dx = touch2.clientX - touch1.clientX;
        const dy = touch2.clientY - touch1.clientY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        pinchStartDistanceRef.current = distance;
        pinchStartZoomRef.current = zoom;
        pinchStartZoomCenterRef.current = { ...zoomCenter }; // Store current zoom center
        pinchCenterRef.current = {
            x: (touch1.clientX + touch2.clientX) / 2,
            y: (touch1.clientY + touch2.clientY) / 2
        };
    }, [zoomPanEnabled, zoom, zoomCenter]);

    const handlePinchTouchMove = useCallback((e) => {
        if (!zoomPanEnabled) return;
        if (e.touches.length !== 2) return;
        if (pinchStartDistanceRef.current === null) return;
        if (!containerRef.current) return;

        e.preventDefault();

        const rect = containerRef.current.getBoundingClientRect();

        // Calculate current distance between two fingers
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const dx = touch2.clientX - touch1.clientX;
        const dy = touch2.clientY - touch1.clientY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Calculate zoom based on pinch gesture
        const scale = distance / pinchStartDistanceRef.current;
        const newZoom = Math.max(1, Math.min(10, pinchStartZoomRef.current * scale));

        // Calculate current center of pinch (midpoint between fingers)
        const currentCenterX = (touch1.clientX + touch2.clientX) / 2;
        const currentCenterY = (touch1.clientY + touch2.clientY) / 2;

        // Calculate how much the pinch center has moved in screen pixels
        const screenDeltaX = currentCenterX - pinchCenterRef.current.x;
        const screenDeltaY = currentCenterY - pinchCenterRef.current.y;

        // Convert screen delta to normalized delta.
        // The transform is: screen = zoomCenter + (image - zoomCenter) * zoom
        // Solving for the new zoomCenter that keeps the same image point under the pinch center:
        //   c' = c - Δscreen / (zoom - 1)
        const normalizedDeltaX = (screenDeltaX / rect.width) / (newZoom - 1);
        const normalizedDeltaY = (screenDeltaY / rect.height) / (newZoom - 1);

        // Calculate new zoom center by subtracting the delta from the start position
        // Subtract because moving fingers right should move the view left (pan right)
        let newZoomCenterX = pinchStartZoomCenterRef.current.x - normalizedDeltaX;
        let newZoomCenterY = pinchStartZoomCenterRef.current.y - normalizedDeltaY;

        // Clamp zoom center to valid range [0, 1] to allow panning to image edges
        newZoomCenterX = Math.max(0, Math.min(1, newZoomCenterX));
        newZoomCenterY = Math.max(0, Math.min(1, newZoomCenterY));

        setZoom(newZoom);
        setZoomCenter({ x: newZoomCenterX, y: newZoomCenterY });
    }, [zoomPanEnabled]);

    const handlePinchTouchEnd = useCallback((e) => {
        if (!zoomPanEnabled) return;
        // Reset pinch state when fingers are lifted
        if (e.touches.length < 2) {
            pinchStartDistanceRef.current = null;
        }
    }, [zoomPanEnabled]);

    // Handle middle mouse button pan start
    const handlePanMouseDown = useCallback((e) => {
        if (!zoomPanEnabled) return;
        if (e.button !== 1) return; // Only middle mouse button

        // Prevent browser's auto-scroll behavior (scroll anchor)
        e.preventDefault();
        e.stopPropagation();
        
        setIsPanning(true);
        panStartRef.current = { x: e.clientX, y: e.clientY };
        panStartZoomCenterRef.current = { ...zoomCenter };
    }, [zoomPanEnabled, zoomCenter]);

    // Handle middle mouse button pan move (works globally)
    const handlePanMouseMoveGlobal = useCallback((e) => {
        if (!containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();

        // Calculate how much the mouse has moved in screen pixels
        const screenDeltaX = e.clientX - panStartRef.current.x;
        const screenDeltaY = e.clientY - panStartRef.current.y;

        // Convert screen delta to normalized delta.
        // The transform is: screen = zoomCenter + (image - zoomCenter) * zoom
        // Solving for the new zoomCenter that keeps the same image point under the cursor:
        //   c' = c - Δscreen / (zoom - 1)
        const normalizedDeltaX = (screenDeltaX / rect.width) / (zoom - 1);
        const normalizedDeltaY = (screenDeltaY / rect.height) / (zoom - 1);

        // Calculate new zoom center by subtracting the delta from the start position
        let newZoomCenterX = panStartZoomCenterRef.current.x - normalizedDeltaX;
        let newZoomCenterY = panStartZoomCenterRef.current.y - normalizedDeltaY;

        // Clamp zoom center to valid range [0, 1] to allow panning to image edges
        newZoomCenterX = Math.max(0, Math.min(1, newZoomCenterX));
        newZoomCenterY = Math.max(0, Math.min(1, newZoomCenterY));

        setZoomCenter({ x: newZoomCenterX, y: newZoomCenterY });
    }, [zoom]);

    // Handle middle mouse button pan end (works globally)
    const handlePanMouseUpGlobal = useCallback((e) => {
        // End panning on middle button release or any button release (for mouseup without button info)
        if (e.button === 1 || e.button === undefined) {
            setIsPanning(false);
        }
    }, []);

    // Add/remove global event listeners for panning (allows panning even when mouse leaves the component)
    useEffect(() => {
        if (!isPanning) return;

        const handleMouseMove = (e) => {
            handlePanMouseMoveGlobal(e);
        };

        const handleMouseUp = (e) => {
            handlePanMouseUpGlobal(e);
        };

        // Add global listeners
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isPanning, handlePanMouseMoveGlobal, handlePanMouseUpGlobal]);

    // Handle wheel zoom with native listener (non-passive to allow preventDefault)
    // This replaces the React onWheel handler to ensure scroll prevention works
    useEffect(() => {
        if (!zoomPanEnabled) return;

        const container = containerRef.current;
        if (!container) return;

        const handleWheelNative = (e) => {
            // Call the latest wheel handler via ref
            if (wheelHandlerRef.current) {
                wheelHandlerRef.current(e);
            }
        };

        // Add non-passive wheel listener to prevent default scroll and handle zoom
        container.addEventListener('wheel', handleWheelNative, { passive: false });

        return () => {
            container.removeEventListener('wheel', handleWheelNative);
        };
    }, [zoomPanEnabled]);

    // Handle click
    const handleClick = (e) => {
        if (mode !== 'designer') return;
        if (src && !imageLoaded) return;

        const screenX = e.clientX;
        const screenY = e.clientY;
        const normCoords = screenToNormalized(screenX, screenY);

        // If currently drawing a polygon
        if (currentPolygon) {
            // Check if clicking on first point to close (need at least 3 points)
            if (currentPolygon.length >= 3 && isNearPoint(screenX, screenY, currentPolygon[0], 12)) {
                // Close polygon
                const newPolygons = [...polygons, currentPolygon];
                setPolygons(newPolygons);
                setCurrentPolygon(null);
                if (onChange) {
                    onChange(newPolygons);
                }
                return;
            }

            // Add new point if under max
            if (currentPolygon.length < maxPoints) {
                const newCurrentPolygon = [...currentPolygon, normCoords];

                // Auto-close if maxPoints reached
                if (newCurrentPolygon.length >= maxPoints) {
                    const newPolygons = [...polygons, newCurrentPolygon];
                    setPolygons(newPolygons);
                    setCurrentPolygon(null);
                    if (onChange) {
                        onChange(newPolygons);
                    }
                } else {
                    setCurrentPolygon(newCurrentPolygon);
                }
            }
            return;
        }

        // Not drawing - check if clicking on existing point (to start dragging - handled in mousedown)
        const pointAt = findPointAt(screenX, screenY);
        if (pointAt) return;

        // Check if clicking inside existing polygon (don't start new polygon)
        const polygonAt = findPolygonAt(screenX, screenY);
        if (polygonAt !== null) return;

        // Start new polygon
        if (polygons.length < maxPolygons) {
            setCurrentPolygon([normCoords]);
        }
    };

    // Handle mouse/touch down (for dragging in designer mode, or joystick start)
    const handlePointerDown = (e, isTouch = false) => {
        const { clientX, clientY } = getClientCoords(e);

        if (mode === 'joystick') {
            // Cancel any ongoing animation
            if (joystickAnimationRef.current) {
                cancelAnimationFrame(joystickAnimationRef.current);
                joystickAnimationRef.current = null;
            }

            // Reset joystick session state
            joystickReleasedRef.current = false;
            joystickHadMovementRef.current = false;

            const normCoords = screenToNormalized(clientX, clientY);
            setJoystickStatic(normCoords);
            setJoystickDynamic(normCoords);
            setIsJoystickActive(true);

            if (onJoystickStart) {
                onJoystickStart();
            }

            e.preventDefault();
            return;
        }

        if (mode !== 'designer') return;
        if (src && !imageLoaded) return;
        if (currentPolygon) return; // Don't drag while drawing

        const pointAt = findPointAt(clientX, clientY);
        if (pointAt && pointAt.polyIndex >= 0) {
            setDraggingPoint(pointAt);
            e.preventDefault();
        }
    };

    const handleMouseDown = (e) => handlePointerDown(e, false);
    const handleTouchStart = (e) => handlePointerDown(e, true);

    // Handle mouse/touch move
    const handlePointerMove = (e, isTouch = false) => {
        const { clientX, clientY } = getClientCoords(e);

        if (mode === 'joystick' && isJoystickActive && joystickStatic) {
            // If joystick was released, don't process moves (animation might still be running)
            if (joystickReleasedRef.current) {
                return;
            }
            if (isTouch) e.preventDefault(); // Prevent scrolling while using joystick
            const normCoords = screenToNormalized(clientX, clientY);

            // Calculate max length in normalized coordinates based on reference dimension
            const referenceDim = getJoystickReferenceDimension();
            // We want max length to be constant in screen pixels
            const maxLengthPx = referenceDim * joystickLineMaxLength;

            // Calculate distance from static point
            const dx = normCoords.x - joystickStatic.x;
            const dy = normCoords.y - joystickStatic.y;

            // Convert to pixel space for distance calculation
            // dx is normalized image delta. Need to convert to screen pixels.
            // screenDelta = imageDelta * zoom * containerSize
            const zoomFactor = zoomPanEnabled ? zoom : 1;
            const dxPx = dx * containerSize.width * zoomFactor;
            const dyPx = dy * containerSize.height * zoomFactor;
            const distPx = Math.sqrt(dxPx * dxPx + dyPx * dyPx);

            let clampedCoords = normCoords;
            if (distPx > maxLengthPx) {
                // Clamp to max length
                const scale = maxLengthPx / distPx;
                clampedCoords = {
                    x: joystickStatic.x + dx * scale,
                    y: joystickStatic.y + dy * scale
                };
            }

            setJoystickDynamic(clampedCoords);

            // Notify callback with offset from center (normalized -1 to 1)
            if (onJoystickMove) {
                // offsetX should be fraction of maxLengthPx
                // recalculate distPx for clamped coords
                const dxPxClamped = (clampedCoords.x - joystickStatic.x) * containerSize.width * zoomFactor;
                const dyPxClamped = (clampedCoords.y - joystickStatic.y) * containerSize.height * zoomFactor;
                
                const offsetX = dxPxClamped / maxLengthPx;
                const offsetY = dyPxClamped / maxLengthPx;
                
                // Re-calculate distPx for ratio
                const distPxClamped = Math.sqrt(dxPxClamped * dxPxClamped + dyPxClamped * dyPxClamped);
                const distRatio = distPxClamped / maxLengthPx;

                // Calculate the coords to send
                let coordsToSend;
                if (distRatio <= joystickZeroRadius) {
                    coordsToSend = { x: 0, y: 0 };
                } else {
                    // Invert Y axis: Up (smaller screen Y) should be positive
                    coordsToSend = { x: offsetX, y: -offsetY };
                }

                // Mark that there was movement
                joystickHadMovementRef.current = true;

                // Throttling logic
                if (joystickMoveInterval <= 0) {
                    // No throttling - call immediately
                    onJoystickMove(coordsToSend);
                    lastSentJoystickCoordsRef.current = coordsToSend;
                } else {
                    const now = Date.now();
                    const timeSinceLastCall = now - lastJoystickMoveTimeRef.current;

                    if (timeSinceLastCall >= joystickMoveInterval) {
                        // Interval has passed - call immediately
                        onJoystickMove(coordsToSend);
                        lastJoystickMoveTimeRef.current = now;
                        lastSentJoystickCoordsRef.current = coordsToSend;
                        pendingJoystickMoveRef.current = null;

                        // Clear any pending timeout
                        if (joystickThrottleTimeoutRef.current) {
                            clearTimeout(joystickThrottleTimeoutRef.current);
                            joystickThrottleTimeoutRef.current = null;
                        }
                    } else {
                        // Store pending coords
                        pendingJoystickMoveRef.current = coordsToSend;

                        // Schedule callback if not already scheduled
                        if (!joystickThrottleTimeoutRef.current) {
                            const remainingTime = joystickMoveInterval - timeSinceLastCall;
                            joystickThrottleTimeoutRef.current = setTimeout(() => {
                                joystickThrottleTimeoutRef.current = null;

                                // Don't fire if joystick was released
                                if (joystickReleasedRef.current) {
                                    pendingJoystickMoveRef.current = null;
                                    return;
                                }

                                const pending = pendingJoystickMoveRef.current;
                                const lastSent = lastSentJoystickCoordsRef.current;

                                // Only call if position changed
                                if (pending && (!lastSent || pending.x !== lastSent.x || pending.y !== lastSent.y)) {
                                    onJoystickMove(pending);
                                    lastJoystickMoveTimeRef.current = Date.now();
                                    lastSentJoystickCoordsRef.current = pending;
                                }
                                pendingJoystickMoveRef.current = null;
                            }, remainingTime);
                        }
                    }
                }
            }
            return;
        }

        if (mode !== 'designer') return;
        if (src && !imageLoaded) return;

        if (draggingPoint) {
            if (isTouch) e.preventDefault(); // Prevent scrolling while dragging points
            const normCoords = screenToNormalized(clientX, clientY);
            const newPolygons = [...polygons];
            newPolygons[draggingPoint.polyIndex] = [...newPolygons[draggingPoint.polyIndex]];
            newPolygons[draggingPoint.polyIndex][draggingPoint.pointIndex] = normCoords;
            setPolygons(newPolygons);
            if (onChange) {
                onChange(newPolygons);
            }
        }
    };

    const handleMouseMove = (e) => handlePointerMove(e, false);
    const handleTouchMove = (e) => handlePointerMove(e, true);

    // Handle mouse/touch up
    const handlePointerUp = () => {
        if (mode === 'joystick' && isJoystickActive && joystickStatic && joystickDynamic) {
            // Mark as released immediately to prevent throttle callbacks
            joystickReleasedRef.current = true;

            // Clear throttle timeout
            if (joystickThrottleTimeoutRef.current) {
                clearTimeout(joystickThrottleTimeoutRef.current);
                joystickThrottleTimeoutRef.current = null;
            }

            // Reset throttling refs
            pendingJoystickMoveRef.current = null;
            lastSentJoystickCoordsRef.current = null;
            lastJoystickMoveTimeRef.current = 0;

            // Fire callbacks immediately (before animation)
            // If there was movement, send final (0, 0) position
            if (joystickHadMovementRef.current && onJoystickMove) {
                onJoystickMove({ x: 0, y: 0 });
            }

            // Fire end callback
            if (onJoystickEnd) {
                onJoystickEnd();
            }

            // Animate snap back (visual only, no callbacks at end)
            const startTime = performance.now();
            const startPos = { ...joystickDynamic };
            const endPos = { ...joystickStatic };
            const duration = joystickSnapAnimationDuration * 1000; // ms

            const animate = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(1, elapsed / duration);

                // Ease out
                const eased = 1 - Math.pow(1 - progress, 2);

                const currentPos = {
                    x: startPos.x + (endPos.x - startPos.x) * eased,
                    y: startPos.y + (endPos.y - startPos.y) * eased
                };

                setJoystickDynamic(currentPos);

                if (progress < 1) {
                    joystickAnimationRef.current = requestAnimationFrame(animate);
                } else {
                    // Animation complete, hide joystick
                    setJoystickStatic(null);
                    setJoystickDynamic(null);
                    setIsJoystickActive(false);
                    joystickAnimationRef.current = null;
                }
            };

            joystickAnimationRef.current = requestAnimationFrame(animate);
            return;
        }

        // Only update state if actually dragging (avoid unnecessary state updates)
        if (draggingPoint !== null) {
            setDraggingPoint(null);
        }
    };

    const handleMouseUp = () => handlePointerUp();
    const handleTouchEnd = () => handlePointerUp();

    // Global listener setup for dragging/joystick
    useEffect(() => {
        const handleGlobalMove = (e) => {
            if (isJoystickActive || draggingPoint !== null) {
                handleMouseMove(e);
            }
        };

        const handleGlobalUp = (e) => {
            if (isJoystickActive || draggingPoint !== null) {
                handleMouseUp(e);
            }
        };

        if (isJoystickActive || draggingPoint !== null) {
            window.addEventListener('mousemove', handleGlobalMove);
            window.addEventListener('mouseup', handleGlobalUp);
        }

        return () => {
            window.removeEventListener('mousemove', handleGlobalMove);
            window.removeEventListener('mouseup', handleGlobalUp);
        };
    }, [isJoystickActive, draggingPoint, handleMouseMove, handleMouseUp]);

    // Handle double click (delete polygon)
    const handleDoubleClick = (e) => {
        if (mode !== 'designer') return;
        if (src && !imageLoaded) return;

        // Cancel current drawing
        if (currentPolygon) {
            setCurrentPolygon(null);
            return;
        }

        // Delete polygon at click position
        const polygonAt = findPolygonAt(e.clientX, e.clientY);
        if (polygonAt !== null) {
            const newPolygons = polygons.filter((_, index) => index !== polygonAt);
            setPolygons(newPolygons);
            if (onChange) {
                onChange(newPolygons);
            }
        }
    };

    // Get image style (same as Image component)
    const getImageStyle = () => {
        // Base styles to prevent long-press context menu on mobile
        const preventLongPressStyles = (mode === 'designer' || mode === 'joystick') ? {
            WebkitTouchCallout: 'none', // Disable iOS callout
            WebkitUserSelect: 'none', // Disable selection
            userSelect: 'none',
            pointerEvents: 'none' // Let events pass through to container
        } : {};

        switch (stretchMode) {
            case 'fit':
                return { display: 'block', width: '100%', height: '100%', objectFit: 'contain', ...preventLongPressStyles };
            case 'stretch':
                return { display: 'block', width: '100%', height: '100%', objectFit: 'fill', ...preventLongPressStyles };
            case 'originalSize':
                return { display: 'block', width: 'auto', height: 'auto', maxWidth: '100%', maxHeight: '100%', ...preventLongPressStyles };
            default:
                return { display: 'block', ...preventLongPressStyles };
        }
    };

    const containerStyle = {
        border: border > 0 ? `${border}px dashed gray` : 'none',
        display: stretchMode === 'originalSize' ? 'inline-block' : 'block',
        width: stretchMode === 'originalSize' ? 'auto' : '100%',
        height: stretchMode === 'originalSize' ? 'auto' : '100%',
        backgroundColor: background,
        position: 'relative',
        cursor: isPanning ? 'grabbing' : (mode === 'designer' && currentPolygon ? 'crosshair' : (mode === 'joystick' ? 'pointer' : 'default')),
        touchAction: (mode === 'designer' || mode === 'joystick' || zoomPanEnabled) ? 'none' : 'auto',
        overflow: 'hidden',
        ...style
    };

    // Get transform style for zoom content wrapper
    const getContentTransformStyle = () => {
        if (!zoomPanEnabled || zoom === 1) {
            return {};
        }
        // Transform origin should be at the zoom center point
        const originX = (zoomCenter.x * 100).toFixed(2);
        const originY = (zoomCenter.y * 100).toFixed(2);
        return {
            transform: `scale(${zoom})`,
            transformOrigin: `${originX}% ${originY}%`
        };
    };

    // Update canvas size
    useEffect(() => {
        if (containerRef.current && canvasRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            canvasRef.current.width = rect.width;
            canvasRef.current.height = rect.height;
            draw();
        }
    }, [containerSize, imageLoaded, draw]);

    // Only attach event handlers for interactive modes (designer/joystick)
    const isInteractive = mode === 'designer' || mode === 'joystick';

    return (
        <div
            className="custom-polygon"
            ref={containerRef}
            style={containerStyle}
            onClick={isInteractive ? handleClick : undefined}
            onMouseDown={(e) => {
                if (zoomPanEnabled && e.button === 1) {
                    handlePanMouseDown(e);
                } else if (isInteractive) {
                    handleMouseDown(e);
                }
            }}
            onMouseMove={(e) => {
                // Panning is handled by global listeners when isPanning is true
                if (!isPanning && isInteractive) {
                    handleMouseMove(e);
                }
            }}
            onMouseUp={(e) => {
                // Pan mouseup is handled by global listener
                if (isInteractive) {
                    handleMouseUp(e);
                }
            }}
            onMouseLeave={(e) => {
                // Don't stop panning or joystick/dragging on mouse leave - global listener handles it
                // Only handle non-pan interactive mode events if NOT dragging
                if (!isPanning && isInteractive && !isJoystickActive && draggingPoint === null) {
                    handleMouseUp(e);
                }
            }}
            onTouchStart={(e) => {
                if (zoomPanEnabled && e.touches.length === 2) {
                    handlePinchTouchStart(e);
                } else if (isInteractive) {
                    handleTouchStart(e);
                }
            }}
            onTouchMove={(e) => {
                if (zoomPanEnabled && e.touches.length === 2) {
                    handlePinchTouchMove(e);
                } else if (isInteractive) {
                    handleTouchMove(e);
                }
            }}
            onTouchEnd={(e) => {
                if (zoomPanEnabled) {
                    handlePinchTouchEnd(e);
                }
                if (isInteractive) {
                    handleTouchEnd(e);
                }
            }}
            onDoubleClick={isInteractive ? handleDoubleClick : undefined}
            onAuxClick={(e) => {
                // Prevent browser's auto-scroll behavior when middle-clicking
                if (zoomPanEnabled && e.button === 1) {
                    e.preventDefault();
                }
            }}
        >
            <div
                style={{
                    width: '100%',
                    height: '100%',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    overflow: 'hidden',
                    ...getContentTransformStyle()
                }}
            >
                {src && (
                    <img
                        // Key on `src` forces React to unmount the old
                        // <img> and mount a new one whenever the URL
                        // changes.  This is critical for MJPEG streams:
                        // Chrome (and some other browsers) keep the
                        // previous multipart/x-mixed-replace connection
                        // alive when the `src` attribute is simply
                        // reassigned, which causes ~10s stalls when the
                        // new stream URL is requested (e.g. Manual
                        // Control Save after a mode switch).  Forcing a
                        // full DOM remount releases the old connection
                        // immediately and the browser opens a fresh one
                        // for the new URL.
                        key={src}
                        ref={imageRef}
                        src={src}
                        alt=""
                        style={getImageStyle()}
                        onLoad={handleImageLoad}
                        onError={handleImageLoad}
                        draggable={false}
                    />
                )}
            </div>
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none'
                }}
            />
        </div>
    );
};

export default Polygon;
