import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import MultiSwitch from './MultiSwitch';
import Button from './Button';

const MAX_RENDERER_PIXEL_RATIO = 1.5;
const RENDER_HEARTBEAT_MS = 1000;

const createDisposalContext = () => ({
    geometries: new Set(),
    materials: new Set(),
    textures: new Set()
});

const disposeMaterialTextureValue = (value, disposalContext) => {
    if (!value) {
        return;
    }

    if (Array.isArray(value)) {
        value.forEach((entry) => disposeMaterialTextureValue(entry, disposalContext));
        return;
    }

    if (value.isTexture && !disposalContext.textures.has(value)) {
        value.dispose();
        disposalContext.textures.add(value);
    }
};

const disposeMaterial = (material, disposalContext) => {
    if (Array.isArray(material)) {
        material.forEach((entry) => disposeMaterial(entry, disposalContext));
        return;
    }

    if (!material || disposalContext.materials.has(material)) {
        return;
    }

    Object.values(material).forEach((value) => {
        disposeMaterialTextureValue(value, disposalContext);
    });

    material.dispose();
    disposalContext.materials.add(material);
};

const disposeObjectResources = (object, disposalContext) => {
    if (object.geometry && !disposalContext.geometries.has(object.geometry)) {
        object.geometry.dispose();
        disposalContext.geometries.add(object.geometry);
    }

    if (object.material) {
        disposeMaterial(object.material, disposalContext);
    }
};

const detachAndDisposeObject = (object, disposalContext = createDisposalContext()) => {
    if (!object) {
        return;
    }

    object.traverse((child) => {
        disposeObjectResources(child, disposalContext);
    });

    object.clear();

    if (object.parent) {
        object.parent.remove(object);
    }
};

const clearObjectChildren = (object3D, disposalContext = createDisposalContext()) => {
    if (!object3D) {
        return;
    }

    [...object3D.children].forEach((child) => {
        detachAndDisposeObject(child, disposalContext);
    });
};

const removeNamedSceneObject = (scene, objectName, disposalContext = createDisposalContext()) => {
    if (!scene) {
        return;
    }

    const object = scene.getObjectByName(objectName);
    if (object) {
        detachAndDisposeObject(object, disposalContext);
    }
};

const getRendererPixelRatio = () => {
    if (typeof window === 'undefined') {
        return 1;
    }

    return Math.min(window.devicePixelRatio || 1, MAX_RENDERER_PIXEL_RATIO);
};

const Scene3D = ({
    background = '#ffffffff',
    gridColor = '#eeeeeeff',
    objects = [],
    style = {}
}) => {
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const updateGridRef = useRef(null);
    const cameraRef = useRef(null);
    const objectsGroupRef = useRef(null);
    const lightRef = useRef(null);
    const animationFrameRef = useRef(null);
    const heartbeatIntervalRef = useRef(null);
    const currentGridScaleRef = useRef(1);

    const [controlMode, setControlMode] = useState('move');
    const isDraggingRef = useRef(false);
    const lastMouseRef = useRef({ x: 0, y: 0 });

    // Camera orbit controls state
    const cameraStateRef = useRef({
        distance: 10,
        theta: Math.PI / 4,  // horizontal angle
        phi: Math.PI / 3,    // vertical angle
        target: new THREE.Vector3(0, 0, 0)
    });

    // Parse color with alpha
    const parseColor = (colorString) => {
        if (!colorString) return { color: 0xffffff, opacity: 1 };

        let hex = colorString.replace('#', '');
        let opacity = 1;

        if (hex.length === 8) {
            opacity = parseInt(hex.slice(6, 8), 16) / 255;
            hex = hex.slice(0, 6);
        }

        return { color: parseInt(hex, 16), opacity };
    };

    const renderScene = useCallback(() => {
        const renderer = rendererRef.current;
        const scene = sceneRef.current;
        const camera = cameraRef.current;

        if (!renderer || !scene || !camera) {
            return;
        }

        renderer.render(scene, camera);
    }, []);

    const requestRender = useCallback(() => {
        if (animationFrameRef.current !== null) {
            return;
        }

        if (typeof window === 'undefined' || typeof window.requestAnimationFrame !== 'function') {
            renderScene();
            return;
        }

        animationFrameRef.current = window.requestAnimationFrame(() => {
            animationFrameRef.current = null;
            renderScene();
        });
    }, [renderScene]);

    // Update camera position based on orbit state
    const updateCameraPosition = useCallback((shouldUpdateGrid = true) => {
        const camera = cameraRef.current;
        const state = cameraStateRef.current;
        if (!camera) return;

        // Blender-style: Z is up
        const x = state.target.x + state.distance * Math.sin(state.phi) * Math.cos(state.theta);
        const y = state.target.y + state.distance * Math.sin(state.phi) * Math.sin(state.theta);
        const z = state.target.z + state.distance * Math.cos(state.phi);

        camera.position.set(x, y, z);
        camera.up.set(0, 0, 1);
        camera.lookAt(state.target);

        // Update light to follow camera
        if (lightRef.current) {
            lightRef.current.position.copy(camera.position);
        }

        // Update grid based on new camera distance
        if (shouldUpdateGrid && updateGridRef.current) {
            updateGridRef.current();
        }

        requestRender();
    }, [requestRender]);

    // Set camera to fixed default position (2, 2, 2) looking at (0, 0, 0)
    const setCameraToDefault = useCallback(() => {
        const defaultTarget = new THREE.Vector3(0, 0, 0);
        const defaultPos = new THREE.Vector3(1.5, 1.5, 1.5);

        // Calculate spherical coordinates for (2, 2, 2)
        const distance = defaultPos.distanceTo(defaultTarget); // ~346.41

        // theta = atan2(y, x)
        // Note: In our current setup, x/y plane checks:
        // x = distance * sin(phi) * cos(theta)
        // y = distance * sin(phi) * sin(theta)
        // z = distance * cos(phi)

        const theta = Math.atan2(defaultPos.y, defaultPos.x);

        // phi = acos(z / r)
        const phi = Math.acos(defaultPos.z / distance);

        cameraStateRef.current = {
            distance: distance,
            theta: theta,
            phi: phi,
            target: defaultTarget
        };
        updateCameraPosition();
    }, [updateCameraPosition]);

    // Create scene objects
    const createObjects = useCallback(() => {
        const group = objectsGroupRef.current;
        if (!group) return;

        // Clear existing objects
        clearObjectChildren(group);

        // Create new objects
        objects.forEach(obj => {
            switch (obj.type) {
                case 'point': {
                    const { color: hexColor, opacity } = parseColor(obj.color || '#ff0000ff');
                    const pointSize = obj.width || 0.2; // Width controls point size
                    const spriteMaterial = new THREE.SpriteMaterial({
                        color: hexColor,
                        opacity: opacity,
                        transparent: opacity < 1,
                        sizeAttenuation: true
                    });
                    const sprite = new THREE.Sprite(spriteMaterial);
                    sprite.position.set(obj.x || 0, obj.y || 0, obj.z || 0);
                    sprite.scale.set(pointSize, pointSize, pointSize);
                    group.add(sprite);
                    break;
                }
                case 'line': {
                    const { color: hexColor, opacity } = parseColor(obj.color || '#00ff00ff');
                    const lineWidth = obj.width || 0.02; // Width controls line thickness
                    const start = new THREE.Vector3(obj.x1 || 0, obj.y1 || 0, obj.z1 || 0);
                    const end = new THREE.Vector3(obj.x2 || 0, obj.y2 || 0, obj.z2 || 0);

                    // Use cylinder geometry to create thick lines
                    const direction = new THREE.Vector3().subVectors(end, start);
                    const length = direction.length();
                    const center = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

                    const geometry = new THREE.CylinderGeometry(lineWidth / 2, lineWidth / 2, length, 8, 1);
                    const material = new THREE.MeshBasicMaterial({
                        color: hexColor,
                        opacity: opacity,
                        transparent: opacity < 1
                    });
                    const cylinder = new THREE.Mesh(geometry, material);

                    // Position and rotate cylinder to align with line
                    cylinder.position.copy(center);
                    cylinder.quaternion.setFromUnitVectors(
                        new THREE.Vector3(0, 1, 0),
                        direction.clone().normalize()
                    );

                    group.add(cylinder);
                    break;
                }
                case 'box': {
                    const { color: hexColor, opacity } = parseColor(obj.color || '#0000ffff');
                    const size = obj.width || 1;
                    const geometry = new THREE.BoxGeometry(size, size, size);
                    const material = new THREE.MeshStandardMaterial({
                        color: hexColor,
                        opacity: opacity,
                        transparent: opacity < 1
                    });
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(obj.x || 0, obj.y || 0, obj.z || 0);
                    group.add(mesh);
                    break;
                }
                case 'sphere': {
                    const { color: hexColor, opacity } = parseColor(obj.color || '#ff00ffff');
                    const radius = obj.radius || 0.5;
                    // Low poly sphere (8 segments)
                    const geometry = new THREE.SphereGeometry(radius, 8, 6);
                    const material = new THREE.MeshStandardMaterial({
                        color: hexColor,
                        opacity: opacity,
                        transparent: opacity < 1,
                        flatShading: true
                    });
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(obj.x || 0, obj.y || 0, obj.z || 0);
                    group.add(mesh);
                    break;
                }
                default:
                    console.warn(`Unknown object type: ${obj.type}`);
            }
        });
        requestRender();
    }, [objects, requestRender]);

    // Initialize Three.js scene
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        // Scene
        const scene = new THREE.Scene();
        sceneRef.current = scene;

        // Camera
        const camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        cameraRef.current = camera;

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(getRendererPixelRatio());
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Objects group
        const objectsGroup = new THREE.Group();
        scene.add(objectsGroup);
        objectsGroupRef.current = objectsGroup;

        // Lighting - from camera
        const pointLight = new THREE.PointLight(0xffffff, 1, 0, 0);
        scene.add(pointLight);
        lightRef.current = pointLight;

        // Ambient light for base illumination
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);

        // Initial camera position
        updateCameraPosition();

        heartbeatIntervalRef.current = window.setInterval(() => {
            requestRender();
        }, RENDER_HEARTBEAT_MS);

        // Handle resize
        const handleResize = () => {
            if (!container || !camera || !renderer) return;
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setPixelRatio(getRendererPixelRatio());
            renderer.setSize(width, height);
            requestRender();
        };

        const resizeObserver = new ResizeObserver(handleResize);
        resizeObserver.observe(container);

        // Cleanup
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            if (heartbeatIntervalRef.current !== null) {
                clearInterval(heartbeatIntervalRef.current);
                heartbeatIntervalRef.current = null;
            }
            resizeObserver.disconnect();
            clearObjectChildren(scene);
            scene.background = null;
            renderer.renderLists.dispose();
            renderer.dispose();
            if (typeof renderer.forceContextLoss === 'function') {
                renderer.forceContextLoss();
            }
            if (container.contains(renderer.domElement)) {
                container.removeChild(renderer.domElement);
            }
            rendererRef.current = null;
            sceneRef.current = null;
            cameraRef.current = null;
            objectsGroupRef.current = null;
            lightRef.current = null;
            updateGridRef.current = null;
            isDraggingRef.current = false;
        };
    }, [requestRender, updateCameraPosition]);

    // Update background color
    useEffect(() => {
        const scene = sceneRef.current;
        if (!scene) return;

        const { color } = parseColor(background);
        scene.background = new THREE.Color(color);
        requestRender();
    }, [background, requestRender]);

    // Update grid based on camera distance (Blender-style dynamic scaling)
    const updateGrid = useCallback(() => {
        const scene = sceneRef.current;
        if (!scene) return;

        const distance = cameraStateRef.current.distance;

        // Calculate grid scale based on distance
        // Use powers of 10 for scale jumps (like Blender)
        // Each scale level is 10x the previous
        const logDistance = Math.log10(distance);
        const scaleLevel = Math.floor(logDistance);
        const gridScale = Math.pow(10, scaleLevel - 1);

        // Only update if scale has changed
        if (gridScale === currentGridScaleRef.current) return;
        currentGridScaleRef.current = gridScale;

        // Remove existing grids
        const gridDisposalContext = createDisposalContext();
        removeNamedSceneObject(scene, 'grid', gridDisposalContext);
        removeNamedSceneObject(scene, 'subgrid', gridDisposalContext);

        const { color } = parseColor(gridColor);

        // Main grid: larger divisions
        const mainGridSize = gridScale * 100;
        const mainDivisions = 10;
        const mainGrid = new THREE.GridHelper(mainGridSize, mainDivisions, color, color);
        mainGrid.rotation.x = Math.PI / 2;
        mainGrid.name = 'grid';
        scene.add(mainGrid);

        // Sub-grid: finer divisions (10x more detailed)
        const subGridColor = parseColor(gridColor).color;
        const subGridMaterial = new THREE.LineBasicMaterial({
            color: subGridColor,
            opacity: 0.3,
            transparent: true
        });
        const subGrid = new THREE.GridHelper(mainGridSize, mainDivisions * 10);
        subGrid.rotation.x = Math.PI / 2;
        subGrid.material = subGridMaterial;
        subGrid.name = 'subgrid';
        scene.add(subGrid);
        requestRender();
    }, [gridColor, requestRender]);

    // Store updateGrid reference for use in updateCameraPosition
    useEffect(() => {
        updateGridRef.current = updateGrid;
    }, [updateGrid]);

    // Initial grid setup
    useEffect(() => {
        // Force initial grid creation by resetting scale ref
        currentGridScaleRef.current = -1;
        updateGrid();
    }, [gridColor, updateGrid]);

    const hasInitializedCameraRef = useRef(false);

    // Update objects and reset camera when objects prop changes
    useEffect(() => {
        createObjects();

        // Only reset camera if we haven't initialized it yet and we have objects to focus on
        if (!hasInitializedCameraRef.current && objects.length > 0) {
            setCameraToDefault();
            hasInitializedCameraRef.current = true;
        }
        // If objects become empty, we might want to allow re-initialization next time objects arrive
        if (objects.length === 0) {
            hasInitializedCameraRef.current = false;
        }
    }, [objects, createObjects, setCameraToDefault]);

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

    // Mouse/touch event handlers
    const handlePointerDown = useCallback((e, isTouch = false) => {
        if (isTouch) e.preventDefault(); // Prevent scrolling on touch
        const { clientX, clientY } = getClientCoords(e);
        isDraggingRef.current = true;
        lastMouseRef.current = { x: clientX, y: clientY };
    }, [getClientCoords]);

    const handleMouseDown = useCallback((e) => handlePointerDown(e, false), [handlePointerDown]);
    const handleTouchStart = useCallback((e) => handlePointerDown(e, true), [handlePointerDown]);

    const handlePointerUp = useCallback(() => {
        isDraggingRef.current = false;
    }, []);

    const handleMouseUp = useCallback(() => handlePointerUp(), [handlePointerUp]);
    const handleTouchEnd = useCallback(() => handlePointerUp(), [handlePointerUp]);

    const handlePointerMove = useCallback((e, isTouch = false) => {
        if (!isDraggingRef.current) return;
        if (isTouch) e.preventDefault(); // Prevent scrolling while dragging

        const { clientX, clientY } = getClientCoords(e);
        const deltaX = clientX - lastMouseRef.current.x;
        const deltaY = clientY - lastMouseRef.current.y;
        lastMouseRef.current = { x: clientX, y: clientY };

        const state = cameraStateRef.current;

        // For touch events, always use the control mode
        // For mouse events, check for middle button and shift key
        let mode = controlMode;
        if (!isTouch) {
            const isMiddleButton = e.buttons === 4;
            const isLeftButton = e.buttons === 1;
            const isShiftHeld = e.shiftKey;

            // Middle button behaviors (like Blender)
            if (isMiddleButton) {
                if (isShiftHeld) {
                    mode = 'move';
                } else {
                    mode = 'rotate';
                }
            }
            // Left button uses the control mode
            else if (!isLeftButton) {
                return; // Not a valid button
            }
        }

        if (mode === 'rotate') {
            // Orbit rotation
            state.theta -= deltaX * 0.01;
            state.phi -= deltaY * 0.01;
            state.phi = Math.max(0.1, Math.min(Math.PI - 0.1, state.phi));
        } else if (mode === 'move') {
            // Pan
            const camera = cameraRef.current;
            if (camera) {
                const right = new THREE.Vector3();
                const up = new THREE.Vector3(0, 0, 1);
                camera.getWorldDirection(right);
                right.cross(up).normalize();

                const moveSpeed = state.distance * 0.002;
                state.target.add(right.multiplyScalar(-deltaX * moveSpeed));

                const forward = new THREE.Vector3();
                camera.getWorldDirection(forward);
                forward.z = 0;
                forward.normalize();
                state.target.add(forward.multiplyScalar(deltaY * moveSpeed));
            }
        } else if (mode === 'zoom') {
            // Zoom with drag
            const zoomSpeed = 0.01;
            state.distance *= 1 + deltaY * zoomSpeed;
            state.distance = Math.max(1, Math.min(100, state.distance));
        }

        updateCameraPosition();
    }, [controlMode, updateCameraPosition, getClientCoords]);

    const handleMouseMove = useCallback((e) => handlePointerMove(e, false), [handlePointerMove]);
    const handleTouchMove = useCallback((e) => handlePointerMove(e, true), [handlePointerMove]);

    const handleWheel = useCallback((e) => {
        e.preventDefault();
        const state = cameraStateRef.current;
        const zoomSpeed = 0.001;
        state.distance *= 1 + e.deltaY * zoomSpeed;
        state.distance = Math.max(1, Math.min(100, state.distance));
        updateCameraPosition();
    }, [updateCameraPosition]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) {
            return undefined;
        }

        const handleNonPassiveWheel = (event) => {
            handleWheel(event);
        };

        container.addEventListener('wheel', handleNonPassiveWheel, { passive: false });

        return () => {
            container.removeEventListener('wheel', handleNonPassiveWheel);
        };
    }, [handleWheel]);

    // Handle context menu (prevent on canvas)
    const handleContextMenu = useCallback((e) => {
        e.preventDefault();
    }, []);

    const containerStyle = {
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        touchAction: 'none', // Prevent browser scroll on touch
        ...style
    };

    const controlsStyle = {
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        zIndex: 10
    };

    const resetButtonStyle = {
        position: 'absolute',
        bottom: '10px',
        right: '10px',
        zIndex: 10
    };

    // Match MultiSwitch default colors
    const resetButtonColors = {
        backgroundColor: '#cccccc',
        color: '#333333',
        border: '1px solid #999999',
        borderRadius: '0.375rem',
        padding: '0.5rem 1rem',
        fontWeight: 400,
        opacity: 0.5,
        width: '70px',
        textAlign: 'center'
    };

    return (
        <div
            ref={containerRef}
            style={containerStyle}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onContextMenu={handleContextMenu}
        >
            <div style={controlsStyle}>
                <MultiSwitch
                    options={[
                        { label: 'Move', value: 'move' },
                        { label: 'Rotate', value: 'rotate' },
                        { label: 'Zoom', value: 'zoom' }
                    ]}
                    value={controlMode}
                    onChange={setControlMode}
                    orientation="vertical"
                    style={{ opacity: 0.5 }}
                />
            </div>
            <div style={resetButtonStyle}>
                <Button
                    label="Reset"
                    onClick={setCameraToDefault}
                    style={resetButtonColors}
                />
            </div>
        </div>
    );
};

export default Scene3D;
