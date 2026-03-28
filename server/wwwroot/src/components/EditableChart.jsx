import React, { useState, useRef, useEffect } from 'react';

const EditableChart = ({
    xMin = 0,
    xMax = 100,
    yMin = 0,
    yMax = 100,
    xLabel = 'X',
    yLabel = 'Y',
    maxNumberOfUserPoints = 10,
    startPoint = { x: 0, y: 50 },
    endPoint = { x: 100, y: 50 },
    points = [],
    onChange = () => { }
}) => {
    const [draggingIndex, setDraggingIndex] = useState(null);
    const [selectedIndex, setSelectedIndex] = useState(null);
    const svgRef = useRef(null);

    const width = 600;
    const height = 400;
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Handle keyboard events for deletion
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Delete' && selectedIndex !== null) {
                const newPoints = points.filter((_, i) => i !== selectedIndex);
                onChange(newPoints);
                setSelectedIndex(null);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedIndex, points, onChange]);

    // Convert data coordinates to SVG coordinates
    const toSVGX = (x) => padding + ((x - xMin) / (xMax - xMin)) * chartWidth;
    const toSVGY = (y) => height - padding - ((y - yMin) / (yMax - yMin)) * chartHeight;

    // Convert SVG coordinates to data coordinates
    const toDataX = (svgX) => xMin + ((svgX - padding) / chartWidth) * (xMax - xMin);
    const toDataY = (svgY) => yMin + ((height - padding - svgY) / chartHeight) * (yMax - yMin);

    // Get all points sorted by X
    const getAllPoints = () => {
        return [startPoint, ...points, endPoint].sort((a, b) => a.x - b.x);
    };

    // Handle click to add point
    const handleChartClick = (e) => {
        if (draggingIndex !== null) return;
        if (points.length >= maxNumberOfUserPoints) return;

        const svg = svgRef.current;
        const rect = svg.getBoundingClientRect();
        const svgX = e.clientX - rect.left;
        const svgY = e.clientY - rect.top;

        // Check if click is within chart area
        if (svgX < padding || svgX > width - padding || svgY < padding || svgY > height - padding) {
            return;
        }

        const dataX = toDataX(svgX);
        const dataY = toDataY(svgY);

        // Clamp values
        const clampedX = Math.max(xMin, Math.min(xMax, dataX));
        const clampedY = Math.max(yMin, Math.min(yMax, dataY));

        onChange([...points, { x: clampedX, y: clampedY }]);
        setSelectedIndex(null);
    };

    // Handle point click/drag start
    const handlePointMouseDown = (index, e) => {
        e.stopPropagation();
        setSelectedIndex(index);
        setDraggingIndex(index);
    };

    // Handle point drag
    const handleMouseMove = (e) => {
        if (draggingIndex === null) return;

        const svg = svgRef.current;
        const rect = svg.getBoundingClientRect();
        const svgX = e.clientX - rect.left;
        const svgY = e.clientY - rect.top;

        const dataX = toDataX(svgX);
        const dataY = toDataY(svgY);

        // Clamp values
        const clampedX = Math.max(xMin, Math.min(xMax, dataX));
        const clampedY = Math.max(yMin, Math.min(yMax, dataY));

        const newPoints = [...points];
        newPoints[draggingIndex] = { x: clampedX, y: clampedY };
        onChange(newPoints);
    };

    // Handle point drag end
    const handleMouseUp = () => {
        setDraggingIndex(null);
    };

    // Handle point delete (double-click)
    const handlePointDoubleClick = (index, e) => {
        e.stopPropagation();
        const newPoints = points.filter((_, i) => i !== index);
        onChange(newPoints);
        setSelectedIndex(null);
    };

    // Generate grid lines
    const generateGridLines = () => {
        const gridLines = [];
        const xSteps = 10;
        const ySteps = 10;

        // Vertical grid lines
        for (let i = 0; i <= xSteps; i++) {
            const x = padding + (i / xSteps) * chartWidth;
            gridLines.push(
                <line
                    key={`v-${i}`}
                    x1={x}
                    y1={padding}
                    x2={x}
                    y2={height - padding}
                    stroke="#e0e0e0"
                    strokeWidth="1"
                />
            );
        }

        // Horizontal grid lines
        for (let i = 0; i <= ySteps; i++) {
            const y = padding + (i / ySteps) * chartHeight;
            gridLines.push(
                <line
                    key={`h-${i}`}
                    x1={padding}
                    y1={y}
                    x2={width - padding}
                    y2={y}
                    stroke="#e0e0e0"
                    strokeWidth="1"
                />
            );
        }

        return gridLines;
    };

    // Generate axis labels
    const generateAxisLabels = () => {
        const labels = [];
        const xSteps = 5;
        const ySteps = 5;

        // X-axis labels
        for (let i = 0; i <= xSteps; i++) {
            const value = xMin + (i / xSteps) * (xMax - xMin);
            const x = padding + (i / xSteps) * chartWidth;
            labels.push(
                <text
                    key={`x-${i}`}
                    x={x}
                    y={height - padding + 25}
                    textAnchor="middle"
                    fontSize="12"
                    fill="#666"
                >
                    {value.toFixed(1)}
                </text>
            );
        }

        // Y-axis labels
        for (let i = 0; i <= ySteps; i++) {
            const value = yMin + (i / ySteps) * (yMax - yMin);
            const y = height - padding - (i / ySteps) * chartHeight;
            labels.push(
                <text
                    key={`y-${i}`}
                    x={padding - 10}
                    y={y + 4}
                    textAnchor="end"
                    fontSize="12"
                    fill="#666"
                >
                    {value.toFixed(1)}
                </text>
            );
        }

        return labels;
    };

    // Generate line path
    const generateLinePath = () => {
        const allPoints = getAllPoints();
        if (allPoints.length === 0) return '';

        let path = `M ${toSVGX(allPoints[0].x)} ${toSVGY(allPoints[0].y)}`;
        for (let i = 1; i < allPoints.length; i++) {
            path += ` L ${toSVGX(allPoints[i].x)} ${toSVGY(allPoints[i].y)}`;
        }
        return path;
    };

    return (
        <div style={{ display: 'inline-block', userSelect: 'none' }}>
            <svg
                ref={svgRef}
                width={width}
                height={height}
                style={{ border: '1px solid #ccc', backgroundColor: '#fff', cursor: 'crosshair' }}
                onClick={handleChartClick}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
            >
                {/* Grid lines */}
                {generateGridLines()}

                {/* Axes */}
                <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#333" strokeWidth="2" />
                <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#333" strokeWidth="2" />

                {/* Axis labels */}
                {generateAxisLabels()}

                {/* Axis titles */}
                <text x={width / 2} y={height - 10} textAnchor="middle" fontSize="14" fontWeight="bold" fill="#333">
                    {xLabel}
                </text>
                <text
                    x={20}
                    y={height / 2}
                    textAnchor="middle"
                    fontSize="14"
                    fontWeight="bold"
                    fill="#333"
                    transform={`rotate(-90, 20, ${height / 2})`}
                >
                    {yLabel}
                </text>

                {/* Line */}
                <path d={generateLinePath()} stroke="#3b82f6" strokeWidth="2" fill="none" />

                {/* Start point (fixed) */}
                <circle
                    cx={toSVGX(startPoint.x)}
                    cy={toSVGY(startPoint.y)}
                    r="6"
                    fill="#10b981"
                    stroke="#fff"
                    strokeWidth="2"
                />

                {/* End point (fixed) */}
                <circle
                    cx={toSVGX(endPoint.x)}
                    cy={toSVGY(endPoint.y)}
                    r="6"
                    fill="#10b981"
                    stroke="#fff"
                    strokeWidth="2"
                />

                {/* User points (editable) - black, smaller, blue when selected */}
                {points.map((point, index) => (
                    <circle
                        key={index}
                        cx={toSVGX(point.x)}
                        cy={toSVGY(point.y)}
                        r="6"
                        fill={selectedIndex === index ? "#3b82f6" : "#000"}
                        stroke="#fff"
                        strokeWidth="2"
                        style={{ cursor: 'move' }}
                        onMouseDown={(e) => handlePointMouseDown(index, e)}
                        onDoubleClick={(e) => handlePointDoubleClick(index, e)}
                    />
                ))}
            </svg>
            <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
                <div>Click to add points ({points.length}/{maxNumberOfUserPoints})</div>
                <div>Drag points to move • Double-click or press Del to delete</div>
            </div>
        </div>
    );
};

export default EditableChart;
