/**
 * HotZoneCloudPointsGenerator
 * 
 * Generates 3D cloud points and scene objects for the hot zone visualization
 * by computing intersections of three spheres representing the arm and center pole reach.
 */

/**
 * Computes the intersection points of three spheres.
 * 
 * The algorithm finds points where three spheres intersect by:
 * 1. Finding the circle of intersection between sphere1 and sphere2
 * 2. Finding where sphere3 intersects with that circle
 * 
 * @param {Object} c1 - Center of sphere 1 {x, y, z}
 * @param {Object} c2 - Center of sphere 2 {x, y, z}
 * @param {Object} c3 - Center of sphere 3 {x, y, z}
 * @param {number} r1 - Radius of sphere 1
 * @param {number} r2 - Radius of sphere 2
 * @param {number} r3 - Radius of sphere 3
 * @returns {Array} Array of intersection points [{x, y, z}, ...] or empty array if no intersection
 */
export const computeIntersectionOfThreeSpheres = (c1, c2, c3, r1, r2, r3) => {
    const results = [];

    // Vector from c1 to c2
    const d12 = {
        x: c2.x - c1.x,
        y: c2.y - c1.y,
        z: c2.z - c1.z
    };

    // Distance between c1 and c2
    const dist12 = Math.sqrt(d12.x * d12.x + d12.y * d12.y + d12.z * d12.z);

    if (dist12 < 1e-10) {
        return results; // Centers coincide, no unique solution
    }

    // Check if spheres 1 and 2 intersect
    if (dist12 > r1 + r2 || dist12 < Math.abs(r1 - r2)) {
        return results; // Spheres don't intersect
    }

    // Unit vector from c1 to c2
    const e12 = {
        x: d12.x / dist12,
        y: d12.y / dist12,
        z: d12.z / dist12
    };

    // Distance from c1 to the plane of intersection circle
    const a = (r1 * r1 - r2 * r2 + dist12 * dist12) / (2 * dist12);

    // Radius of the intersection circle
    const h = Math.sqrt(Math.max(0, r1 * r1 - a * a));

    if (h < 1e-10) {
        // Spheres touch at a single point
        const p = {
            x: c1.x + a * e12.x,
            y: c1.y + a * e12.y,
            z: c1.z + a * e12.z
        };

        // Check if this point is on sphere 3
        const distToC3 = Math.sqrt(
            (p.x - c3.x) * (p.x - c3.x) +
            (p.y - c3.y) * (p.y - c3.y) +
            (p.z - c3.z) * (p.z - c3.z)
        );

        if (Math.abs(distToC3 - r3) < 1e-6) {
            results.push(p);
        }
        return results;
    }

    // Center of the intersection circle
    const circleCenter = {
        x: c1.x + a * e12.x,
        y: c1.y + a * e12.y,
        z: c1.z + a * e12.z
    };

    // Find two perpendicular unit vectors in the plane of the circle
    // First, find a vector not parallel to e12
    let temp = { x: 1, y: 0, z: 0 };
    if (Math.abs(e12.x) > 0.9) {
        temp = { x: 0, y: 1, z: 0 };
    }

    // Cross product to get first perpendicular vector
    let u = {
        x: e12.y * temp.z - e12.z * temp.y,
        y: e12.z * temp.x - e12.x * temp.z,
        z: e12.x * temp.y - e12.y * temp.x
    };
    const uLen = Math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = { x: u.x / uLen, y: u.y / uLen, z: u.z / uLen };

    // Cross product to get second perpendicular vector
    const v = {
        x: e12.y * u.z - e12.z * u.y,
        y: e12.z * u.x - e12.x * u.z,
        z: e12.x * u.y - e12.y * u.x
    };

    // Now find intersection with sphere 3
    // Points on the circle: P = circleCenter + h * (cos(θ) * u + sin(θ) * v)
    // We need |P - c3| = r3

    // Vector from c3 to circle center
    const cc3 = {
        x: circleCenter.x - c3.x,
        y: circleCenter.y - c3.y,
        z: circleCenter.z - c3.z
    };

    // Project cc3 onto the circle plane (onto u and v)
    const cc3_u = cc3.x * u.x + cc3.y * u.y + cc3.z * u.z;
    const cc3_v = cc3.x * v.x + cc3.y * v.y + cc3.z * v.z;
    const cc3_e = cc3.x * e12.x + cc3.y * e12.y + cc3.z * e12.z;

    // Distance squared from c3 to circle center along the axis
    const axialDistSq = cc3_e * cc3_e;

    // We need to solve: h²cos²θ + h²sin²θ + 2h(cc3_u·cosθ + cc3_v·sinθ) + |cc3|² = r3²
    // Simplifies to: h² + 2h(cc3_u·cosθ + cc3_v·sinθ) + |cc3|² = r3²
    // Let A = cc3_u, B = cc3_v
    // 2h(A·cosθ + B·sinθ) = r3² - h² - |cc3|²

    const cc3LenSq = cc3.x * cc3.x + cc3.y * cc3.y + cc3.z * cc3.z;
    const rhs = r3 * r3 - h * h - cc3LenSq;

    const A = cc3_u;
    const B = cc3_v;
    const C = rhs / (2 * h);

    // A·cosθ + B·sinθ = C
    // This is: sqrt(A² + B²) · sin(θ + φ) = C, where tan(φ) = A/B

    const AB = Math.sqrt(A * A + B * B);

    if (AB < 1e-10) {
        // c3 is on the axis of the circle
        if (Math.abs(C) < 1e-6) {
            // The entire circle is on sphere 3 - return some sample points
            for (let i = 0; i < 8; i++) {
                const theta = (2 * Math.PI * i) / 8;
                results.push({
                    x: circleCenter.x + h * (Math.cos(theta) * u.x + Math.sin(theta) * v.x),
                    y: circleCenter.y + h * (Math.cos(theta) * u.y + Math.sin(theta) * v.y),
                    z: circleCenter.z + h * (Math.cos(theta) * u.z + Math.sin(theta) * v.z)
                });
            }
        }
        return results;
    }

    const sinVal = C / AB;

    if (Math.abs(sinVal) > 1) {
        return results; // No intersection
    }

    // Two solutions for the angle
    const phi = Math.atan2(A, B);
    const alpha = Math.asin(sinVal);

    const theta1 = alpha - phi;
    const theta2 = Math.PI - alpha - phi;

    // Compute the two intersection points
    const p1 = {
        x: circleCenter.x + h * (Math.cos(theta1) * u.x + Math.sin(theta1) * v.x),
        y: circleCenter.y + h * (Math.cos(theta1) * u.y + Math.sin(theta1) * v.y),
        z: circleCenter.z + h * (Math.cos(theta1) * u.z + Math.sin(theta1) * v.z)
    };

    const p2 = {
        x: circleCenter.x + h * (Math.cos(theta2) * u.x + Math.sin(theta2) * v.x),
        y: circleCenter.y + h * (Math.cos(theta2) * u.y + Math.sin(theta2) * v.y),
        z: circleCenter.z + h * (Math.cos(theta2) * u.z + Math.sin(theta2) * v.z)
    };

    // Verify both points are on all three spheres
    const verifyPoint = (p) => {
        const d1 = Math.sqrt((p.x - c1.x) ** 2 + (p.y - c1.y) ** 2 + (p.z - c1.z) ** 2);
        const d2 = Math.sqrt((p.x - c2.x) ** 2 + (p.y - c2.y) ** 2 + (p.z - c2.z) ** 2);
        const d3 = Math.sqrt((p.x - c3.x) ** 2 + (p.y - c3.y) ** 2 + (p.z - c3.z) ** 2);
        return Math.abs(d1 - r1) < 0.5 && Math.abs(d2 - r2) < 0.5 && Math.abs(d3 - r3) < 0.5;
    };

    if (verifyPoint(p1)) {
        results.push(p1);
    }
    if (verifyPoint(p2)) {
        // Check if p2 is different from p1
        const dist = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2);
        if (dist > 0.1) {
            results.push(p2);
        }
    }

    return results;
};

/**
 * Generates cloud points and scene objects for the hot zone visualization.
 * 
 * @param {Object} settings - Hot zone settings object
 * @param {Object} settings.centerPole - Center pole settings
 * @param {Object} settings.armPoles - Arm poles settings
 * @param {Object} settings.arms - Arms settings
 * @param {number} calculationQuality - Number of iterations for each arm (higher = more points)
 * @returns {Array} Array of Scene3D objects (points, lines, spheres)
 */
export const generateHotZoneSceneObjects = (settings, calculationQuality = 10) => {
    const { centerPole, armPoles, arms } = settings;

    // Scale factor to convert from cm to scene units (divide by 100 to make reasonable scene size)
    const scale = 0.01;

    // Compute max lengths from min + strokeLen (new format takes priority)
    // If strokeLen is defined, use it; otherwise fall back to old maxLength format for backwards compatibility
    const leftArmMaxLength = arms.leftArmStrokeLen !== undefined
        ? (arms.leftArmMinLength + arms.leftArmStrokeLen)
        : (arms.leftArmMaxLength ?? (arms.leftArmMinLength + 100));
    const rightArmMaxLength = arms.rightArmStrokeLen !== undefined
        ? (arms.rightArmMinLength + arms.rightArmStrokeLen)
        : (arms.rightArmMaxLength ?? (arms.rightArmMinLength + 100));

    // Calculate sphere center points
    const leftSphereCenterPoint = {
        x: -(armPoles.xDistance / 2) * scale,
        y: 0,
        z: armPoles.height * scale
    };

    const rightSphereCenterPoint = {
        x: (armPoles.xDistance / 2) * scale,
        y: 0,
        z: armPoles.height * scale
    };

    const centerSphereCenterPoint = {
        x: 0,
        y: - (centerPole.yDistanceFromArmPoles * scale),
        z: centerPole.height * scale
    };

    const sceneObjects = [];

    // Boundary Points Calculation
    const getLowestIntersection = (rLeft, rRight, rCenter) => {
        const points = computeIntersectionOfThreeSpheres(
            leftSphereCenterPoint,
            rightSphereCenterPoint,
            centerSphereCenterPoint,
            rLeft * scale,
            rRight * scale,
            rCenter * scale
        );

        if (points.length === 0) return null;
        if (points.length === 1) return points[0];

        // Return point with smallest z coordinate (closest to ground)
        return points.reduce((prev, curr) => (prev.z < curr.z ? prev : curr));
    };

    const rMicStick = centerPole.micStickRadius;

    // 1. Top Boundary Point
    const topBoundaryPoint = getLowestIntersection(arms.leftArmMinLength, arms.rightArmMinLength, rMicStick);

    // 2. Left Boundary Point
    const leftBoundaryPoint = getLowestIntersection(arms.leftArmMinLength, rightArmMaxLength, rMicStick);

    // 3. Bottom Boundary Point (Note: Logic in request says Left Max, Right Max)
    const bottomBoundaryPoint = getLowestIntersection(leftArmMaxLength, rightArmMaxLength, rMicStick);

    // 4. Right Boundary Point
    const rightBoundaryPoint = getLowestIntersection(leftArmMaxLength, arms.rightArmMinLength, rMicStick);

    // Helper to check if point is above plane defined by 3 points
    // Returns true if point is on the side of the plane that the normal points to (assuming normal points generally 'up')
    const isPointAbovePlane = (p, v1, v2, v3) => {
        if (!p || !v1 || !v2 || !v3) return false;

        // Vectors for plane
        const u = { x: v2.x - v1.x, y: v2.y - v1.y, z: v2.z - v1.z };
        const v = { x: v3.x - v1.x, y: v3.y - v1.y, z: v3.z - v1.z };

        // Normal vector (cross product)
        let n = {
            x: u.y * v.z - u.z * v.y,
            y: u.z * v.x - u.x * v.z,
            z: u.x * v.y - u.y * v.x
        };

        // Ensure normal points generally upwards (positive z)
        const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len === 0) return false;

        if (n.z < 0) {
            n.x = -n.x;
            n.y = -n.y;
            n.z = -n.z;
        }

        // Dot(N, P - V1)
        const dot = n.x * (p.x - v1.x) + n.y * (p.y - v1.y) + n.z * (p.z - v1.z);

        return dot > 0.0001;
    };

    // Generate cloud points
    for (let i1 = 0; i1 <= calculationQuality; i1++) {
        for (let i2 = 0; i2 <= calculationQuality; i2++) {
            const sphereLeftRadius = (arms.leftArmMinLength +
                (leftArmMaxLength - arms.leftArmMinLength) * i1 / calculationQuality) * scale;
            const sphereRightRadius = (arms.rightArmMinLength +
                (rightArmMaxLength - arms.rightArmMinLength) * i2 / calculationQuality) * scale;
            const sphereCenterRadius = centerPole.micStickRadius * scale;

            const intersections = computeIntersectionOfThreeSpheres(
                leftSphereCenterPoint,
                rightSphereCenterPoint,
                centerSphereCenterPoint,
                sphereLeftRadius,
                sphereRightRadius,
                sphereCenterRadius
            );

            intersections.forEach(point => {
                let color = '#0088ffff'; // Default Blue

                if (point.z < 0) {
                    color = '#ff0000ff'; // Red if below ground
                } else {
                    // Check if above planes
                    // Plane 1: Top, Center, Right
                    // Plane 2: Top, Left, Center
                    // User Request: "above the plane defined by topBoundaryPoint, centerSphereCenterPoint, rightBoundaryPoint"
                    //               "OR above the plane defined by topBoundaryPoint, leftBoundaryPoint and centerSphereCenterPoint"
                    const abovePlane1 = isPointAbovePlane(point, topBoundaryPoint, centerSphereCenterPoint, rightBoundaryPoint);
                    const abovePlane2 = isPointAbovePlane(point, topBoundaryPoint, leftBoundaryPoint, centerSphereCenterPoint);

                    if (abovePlane1 || abovePlane2) {
                        color = '#bbbb00ff'; // Yellowish
                    }
                }

                sceneObjects.push({
                    type: 'point',
                    x: point.x,
                    y: point.y,
                    z: point.z,
                    color: color,
                    width: 0.01
                });
            });
        }
    }

    // Helper to add line
    const addBoundaryLine = (p1, p2, color) => {
        if (!p1 || !p2) return;
        sceneObjects.push({
            type: 'line',
            x1: p1.x,
            y1: p1.y,
            z1: p1.z,
            x2: p2.x,
            y2: p2.y,
            z2: p2.z,
            color: color,
            width: 0.005
        });
    };

    // 1. Red from Left Sphere Center to Top Boundary
    addBoundaryLine(leftSphereCenterPoint, topBoundaryPoint, '#ff0000ff');

    // 2. Green from Right Sphere Center to Top Boundary
    addBoundaryLine(rightSphereCenterPoint, topBoundaryPoint, '#00ff00ff');

    // 3. Blue from Center Sphere Center to Top Boundary
    addBoundaryLine(centerSphereCenterPoint, topBoundaryPoint, '#0000ffff');

    // 4. Blue from Center Sphere Center to Left Boundary
    addBoundaryLine(centerSphereCenterPoint, leftBoundaryPoint, '#0000ffff');

    // 5. Blue from Center Sphere Center to Bottom Boundary
    addBoundaryLine(centerSphereCenterPoint, bottomBoundaryPoint, '#0000ffff');

    // 6. Blue from Center Sphere Center to Right Boundary
    addBoundaryLine(centerSphereCenterPoint, rightBoundaryPoint, '#0000ffff');

    // Add poles as black lines with width 2
    const poleWidth = 0.02;
    const poleColor = '#000000ff';

    // Left pole
    sceneObjects.push({
        type: 'line',
        x1: leftSphereCenterPoint.x,
        y1: leftSphereCenterPoint.y,
        z1: 0,
        x2: leftSphereCenterPoint.x,
        y2: leftSphereCenterPoint.y,
        z2: leftSphereCenterPoint.z,
        color: poleColor,
        width: poleWidth
    });

    // Right pole
    sceneObjects.push({
        type: 'line',
        x1: rightSphereCenterPoint.x,
        y1: rightSphereCenterPoint.y,
        z1: 0,
        x2: rightSphereCenterPoint.x,
        y2: rightSphereCenterPoint.y,
        z2: rightSphereCenterPoint.z,
        color: poleColor,
        width: poleWidth
    });

    // Center pole
    sceneObjects.push({
        type: 'line',
        x1: centerSphereCenterPoint.x,
        y1: centerSphereCenterPoint.y,
        z1: 0,
        x2: centerSphereCenterPoint.x,
        y2: centerSphereCenterPoint.y,
        z2: centerSphereCenterPoint.z,
        color: poleColor,
        width: poleWidth
    });

    // Add anchor spheres on top of each pole
    const anchorSphereRadius = 0.04;

    // Left anchor sphere (red)
    sceneObjects.push({
        type: 'sphere',
        x: leftSphereCenterPoint.x,
        y: leftSphereCenterPoint.y,
        z: leftSphereCenterPoint.z,
        radius: anchorSphereRadius,
        color: '#ff0000ff'
    });

    // Right anchor sphere (green)
    sceneObjects.push({
        type: 'sphere',
        x: rightSphereCenterPoint.x,
        y: rightSphereCenterPoint.y,
        z: rightSphereCenterPoint.z,
        radius: anchorSphereRadius,
        color: '#00ff00ff'
    });

    // Center anchor sphere (blue)
    sceneObjects.push({
        type: 'sphere',
        x: centerSphereCenterPoint.x,
        y: centerSphereCenterPoint.y,
        z: centerSphereCenterPoint.z,
        radius: anchorSphereRadius,
        color: '#0000ffff'
    });

    // FOV Calculation
    const calculateAngle = (p1, center, p2) => {
        if (!p1 || !center || !p2) return 0;

        // Vectors from center to points
        const v1 = { x: p1.x - center.x, y: p1.y - center.y, z: p1.z - center.z };
        const v2 = { x: p2.x - center.x, y: p2.y - center.y, z: p2.z - center.z };

        // Dot product
        const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;

        // Magnitudes
        const m1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
        const m2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

        if (m1 === 0 || m2 === 0) return 0;

        // Cos theta
        const cosTheta = dot / (m1 * m2);

        // Clamp for safety
        const clampedCos = Math.max(-1, Math.min(1, cosTheta));

        // Angle in radians
        const radians = Math.acos(clampedCos);

        // Convert to degrees
        return radians * (180 / Math.PI);
    };

    const horizontalFOV = calculateAngle(leftBoundaryPoint, centerSphereCenterPoint, rightBoundaryPoint);
    const verticalFOV = calculateAngle(topBoundaryPoint, centerSphereCenterPoint, bottomBoundaryPoint);

    return {
        objects: sceneObjects,
        fov: {
            horizontal: horizontalFOV,
            vertical: verticalFOV
        }
    };
};

/**
 * Validates the hot zone settings by checking if all 4 boundary points exist.
 * 
 * @param {Object} settings - Hot zone settings object
 * @returns {boolean} - True if valid (all 4 points exist), False otherwise
 */
export const validateHotZoneSettings = (settings) => {
    const { centerPole, armPoles, arms } = settings;
    const scale = 0.01;

    // Compute max lengths from min + strokeLen (new format takes priority)
    // If strokeLen is defined, use it; otherwise fall back to old maxLength format for backwards compatibility
    const leftArmMaxLength = arms.leftArmStrokeLen !== undefined
        ? (arms.leftArmMinLength + arms.leftArmStrokeLen)
        : (arms.leftArmMaxLength ?? (arms.leftArmMinLength + 100));
    const rightArmMaxLength = arms.rightArmStrokeLen !== undefined
        ? (arms.rightArmMinLength + arms.rightArmStrokeLen)
        : (arms.rightArmMaxLength ?? (arms.rightArmMinLength + 100));

    // Calculate sphere center points
    const leftSphereCenterPoint = {
        x: -(armPoles.xDistance / 2) * scale,
        y: 0,
        z: armPoles.height * scale
    };

    const rightSphereCenterPoint = {
        x: (armPoles.xDistance / 2) * scale,
        y: 0,
        z: armPoles.height * scale
    };

    const centerSphereCenterPoint = {
        x: 0,
        y: - (centerPole.yDistanceFromArmPoles * scale),
        z: centerPole.height * scale
    };

    const rMicStick = centerPole.micStickRadius;

    // Helper to check existence
    const checkIntersection = (rLeft, rRight, rCenter) => {
        const points = computeIntersectionOfThreeSpheres(
            leftSphereCenterPoint,
            rightSphereCenterPoint,
            centerSphereCenterPoint,
            rLeft * scale,
            rRight * scale,
            rCenter * scale
        );
        return points.length > 0;
    };

    // Check all 4 boundary points
    const hasTop = checkIntersection(arms.leftArmMinLength, arms.rightArmMinLength, rMicStick);
    const hasLeft = checkIntersection(leftArmMaxLength, arms.rightArmMinLength, rMicStick);
    const hasBottom = checkIntersection(arms.leftArmMinLength, rightArmMaxLength, rMicStick);
    const hasRight = checkIntersection(leftArmMaxLength, rightArmMaxLength, rMicStick);

    return hasTop && hasLeft && hasBottom && hasRight;
};

export default {
    computeIntersectionOfThreeSpheres,
    generateHotZoneSceneObjects,
    validateHotZoneSettings
};
