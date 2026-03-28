
import { computeIntersectionOfThreeSpheres, generateHotZoneSceneObjects } from './HotZoneCloudPointsGenerator.js';

console.log("Starting HotZone Test Harness - Coloring Verification");

const mockSettings = {
    centerPole: {
        micStickRadius: 100, // 100 cm - increased to reach
        yDistanceFromArmPoles: 50, // 50 cm - closer
        height: 200 // 200 cm
    },
    armPoles: {
        xDistance: 200, // 200 cm
        height: 200 // 200 cm
    },
    arms: {
        leftArmMinLength: 150, // 1.5m - longer to overlap well
        leftArmMaxLength: 300,
        rightArmMinLength: 150, // 1.5m
        rightArmMaxLength: 300
    }
};

// Generate with sufficient quality
const objects = generateHotZoneSceneObjects(mockSettings, 20);

console.log(`Total scene objects: ${objects.length}`);

const points = objects.filter(o => o.type === 'point');
console.log(`Total cloud points: ${points.length}`);

// Analysis
let redPoints = 0;
let yellowPoints = 0;
let bluePoints = 0;

points.forEach(p => {
    if (p.color === '#ff0000ff') {
        redPoints++;
    } else if (p.color === '#bbbb00ff') {
        yellowPoints++;
    } else if (p.color === '#0088ffff') {
        bluePoints++;
    } else {
        console.warn("Unknown color point:", p);
    }
});

console.log(`Red Points (z < 0): ${redPoints}`);
console.log(`Yellow Points (Above Planes): ${yellowPoints}`);
console.log(`Blue Points (Default): ${bluePoints}`);

if (redPoints > 0) console.log("SUCCESS: Red points generation verified.");
else console.warn("WARNING: No red points found. Check if points go below z=0.");

if (yellowPoints > 0) console.log("SUCCESS: Yellow points generation verified.");
else console.warn("WARNING: No yellow points found. Check plane logic.");

// Basic check for boundary lines
const lines = objects.filter(o => o.type === 'line' && o.width === 0.005);
console.log(`Boundary lines count: ${lines.length} (Expect 6)`);
