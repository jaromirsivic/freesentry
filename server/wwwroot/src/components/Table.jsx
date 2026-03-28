import React, { useState, useRef, useEffect, useCallback } from 'react';
import PopupMenu from './PopupMenu';

const Table = ({
    width = '100%',
    height = 400,
    canAddRows = false,
    canAddColumns = false,
    canDeleteRows = false,
    canDeleteColumns = false,
    maxColumns = null,
    maxRows = null,
    minRows = 0,
    cellsEditable = true,
    columnsHeaders: initialColumnsHeaders = null,
    rowsHeaders: initialRowsHeaders = null,
    cells: initialCells = [],
    onCellsChange = null,
    onColumnsHeadersChange = null,
    onRowsHeadersChange = null,
    // New event props
    onChange = null,
    onSelectionChange = null,
    onAddColumn = null,
    onAddRow = null,
    onDeleteColumn = null,
    onDeleteRow = null,
    onResizeColumn = null,
    onResizeRow = null,
    onCopy = null,
    onCut = null,
    onPaste = null
}) => {
    // State
    const [cells, setCells] = useState(() => {
        // Deep clone initial cells
        return initialCells.map(row => row.map(cell => ({ ...cell })));
    });
    const [columnsHeaders, setColumnsHeaders] = useState(() => {
        return initialColumnsHeaders ? initialColumnsHeaders.map(h => ({ ...h })) : null;
    });
    const [rowsHeaders, setRowsHeaders] = useState(() => {
        return initialRowsHeaders ? initialRowsHeaders.map(h => ({ ...h })) : null;
    });
    const [selection, setSelection] = useState({ start: null, end: null });
    const [editingCell, setEditingCell] = useState(null);
    const [editValue, setEditValue] = useState('');
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState(null);
    const [dragEnd, setDragEnd] = useState(null);
    const [isFillDragging, setIsFillDragging] = useState(false);
    const [fillDragStart, setFillDragStart] = useState(null);
    const [fillDragEnd, setFillDragEnd] = useState(null);
    const [history, setHistory] = useState([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [clipboard, setClipboard] = useState(null);
    const [contextMenu, setContextMenu] = useState({ visible: false, x: 0, y: 0 });

    // Column/row resize state
    const [isResizingCol, setIsResizingCol] = useState(false);
    const [resizingColIndex, setResizingColIndex] = useState(null);
    const [resizeColStartX, setResizeColStartX] = useState(0);
    const [resizeColStartWidth, setResizeColStartWidth] = useState(0);
    const [isResizingRow, setIsResizingRow] = useState(false);
    const [resizingRowIndex, setResizingRowIndex] = useState(null);
    const [resizeRowStartY, setResizeRowStartY] = useState(0);
    const [resizeRowStartHeight, setResizeRowStartHeight] = useState(0);

    const tableRef = useRef(null);
    const inputRef  = useRef(null);
    const scrollContainerRef = useRef(null);
    const headerScrollRef = useRef(null);

    // Default column width and row height
    const defaultColWidth = 100;
    const defaultRowHeight = 32;
    const headerRowHeight = 36;
    const headerColWidth = 50;

    // Calculate dimensions
    const numCols = columnsHeaders ? columnsHeaders.length : (cells[0]?.length || 0);
    const numRows = rowsHeaders ? rowsHeaders.length : cells.length;

    // Get column width with min/max constraints
    const getColWidth = (colIndex) => {
        if (columnsHeaders && columnsHeaders[colIndex]?.width) {
            return columnsHeaders[colIndex].width;
        }
        return defaultColWidth;
    };

    // Get row height with min/max constraints
    const getRowHeight = (rowIndex) => {
        if (rowsHeaders && rowsHeaders[rowIndex]?.height) {
            return rowsHeaders[rowIndex].height;
        }
        return defaultRowHeight;
    };

    // Check if column can be resized
    const canResizeColumn = (colIndex) => {
        if (!columnsHeaders || !columnsHeaders[colIndex]) return false;
        return columnsHeaders[colIndex].canResize === true;
    };

    // Check if row can be resized
    const canResizeRow = (rowIndex) => {
        if (!rowsHeaders || !rowsHeaders[rowIndex]) return false;
        return rowsHeaders[rowIndex].canResize === true;
    };

    // Update column headers and notify parent
    const updateColumnsHeaders = useCallback((newHeaders) => {
        setColumnsHeaders(newHeaders);
        if (onColumnsHeadersChange) {
            onColumnsHeadersChange(newHeaders);
        }
        if (onChange) {
            onChange({ type: 'columnsHeaders', data: newHeaders });
        }
    }, [onColumnsHeadersChange, onChange]);

    // Update row headers and notify parent
    const updateRowsHeaders = useCallback((newHeaders) => {
        setRowsHeaders(newHeaders);
        if (onRowsHeadersChange) {
            onRowsHeadersChange(newHeaders);
        }
        if (onChange) {
            onChange({ type: 'rowsHeaders', data: newHeaders });
        }
    }, [onRowsHeadersChange, onChange]);

    // Save state to history
    const saveToHistory = useCallback((newCells) => {
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push(JSON.parse(JSON.stringify(newCells)));
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
    }, [history, historyIndex]);

    // Update cells and notify parent
    const updateCells = useCallback((newCells, saveHistory = true) => {
        if (saveHistory) {
            saveToHistory(newCells);
        }
        setCells(newCells);
        if (onCellsChange) {
            onCellsChange(newCells);
        }
        if (onChange) {
            onChange({ type: 'cells', data: newCells });
        }
    }, [onCellsChange, onChange, saveToHistory]);

    // Undo
    const undo = useCallback(() => {
        if (historyIndex > 0) {
            setHistoryIndex(historyIndex - 1);
            setCells(JSON.parse(JSON.stringify(history[historyIndex - 1])));
        }
    }, [history, historyIndex]);

    // Redo
    const redo = useCallback(() => {
        if (historyIndex < history.length - 1) {
            setHistoryIndex(historyIndex + 1);
            setCells(JSON.parse(JSON.stringify(history[historyIndex + 1])));
        }
    }, [history, historyIndex]);

    // Get cell value
    const getCellValue = (row, col) => {
        if (cells[row] && cells[row][col]) {
            return cells[row][col].value ?? '';
        }
        return '';
    };

    // Get cell style
    const getCellStyle = (row, col) => {
        const cell = cells[row]?.[col] || {};
        return {
            textAlign: cell.align || 'left',
            backgroundColor: cell.backgroundColor || '#ffffff',
            color: cell.textColor || '#000000',
            fontSize: cell.fontSize || '14px',
            fontFamily: cell.fontFamily || 'inherit',
            fontStyle: cell.fontStyle || 'normal',
            fontWeight: cell.fontWeight || 'normal'
        };
    };

    // Check if cell is editable
    const isCellEditable = (row, col) => {
        if (!cellsEditable) return false;
        const cell = cells[row]?.[col];
        if (cell && cell.editable === false) return false;
        return true;
    };

    // Get selection bounds
    const getSelectionBounds = () => {
        if (!selection.start) return null;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);
        return { minRow, maxRow, minCol, maxCol };
    };

    // Check if any selected cell is editable
    const isAnySelectedCellEditable = () => {
        const bounds = getSelectionBounds();
        if (!bounds) return false;
        for (let r = bounds.minRow; r <= bounds.maxRow; r++) {
            for (let c = bounds.minCol; c <= bounds.maxCol; c++) {
                if (isCellEditable(r, c)) return true;
            }
        }
        return false;
    };

    // Check if cell is in selection
    const isCellSelected = (row, col) => {
        if (!selection.start) return false;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);
        return row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
    };

    // Check if cell is in fill drag area
    const isCellInFillArea = (row, col) => {
        if (!fillDragStart || !fillDragEnd) return false;
        const minRow = Math.min(fillDragStart.row, fillDragEnd.row);
        const maxRow = Math.max(fillDragStart.row, fillDragEnd.row);
        const minCol = Math.min(fillDragStart.col, fillDragEnd.col);
        const maxCol = Math.max(fillDragStart.col, fillDragEnd.col);
        return row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
    };

    // Update selection and trigger event
    const updateSelection = useCallback((newSelection) => {
        setSelection(newSelection);
        if (onSelectionChange) {
            onSelectionChange(newSelection);
        }
    }, [onSelectionChange]);

    // Handle cell click
    const handleCellClick = (row, col, e) => {
        if (e.shiftKey && selection.start) {
            updateSelection({ start: selection.start, end: { row, col } });
        } else {
            updateSelection({ start: { row, col }, end: { row, col } });
        }
        setEditingCell(null);
        setContextMenu({ visible: false, x: 0, y: 0 });
    };

    // Handle cell double click
    const handleCellDoubleClick = (row, col) => {
        if (isCellEditable(row, col)) {
            setEditingCell({ row, col });
            setEditValue(getCellValue(row, col));
        }
    };

    // Handle mouse down for selection
    const handleMouseDown = (row, col, e) => {
        if (e.button !== 0) return;
        setIsDragging(true);
        setDragStart({ row, col });
        setDragEnd({ row, col });
        if (!e.shiftKey) {
            updateSelection({ start: { row, col }, end: { row, col } });
        }
    };

    // Handle mouse move for selection
    const handleMouseMove = (row, col) => {
        if (isDragging) {
            setDragEnd({ row, col });
            updateSelection({ start: dragStart, end: { row, col } });
        }
        if (isFillDragging) {
            setFillDragEnd({ row, col });
        }
    };

    // Handle mouse up
    const handleMouseUp = () => {
        if (isFillDragging && fillDragStart && fillDragEnd) {
            // Apply fill
            applyFill();
        }
        setIsDragging(false);
        setIsFillDragging(false);
        setFillDragStart(null);
        setFillDragEnd(null);
    };

    // Column resize handlers
    const handleColumnResizeMouseDown = (colIndex, e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!canResizeColumn(colIndex)) return;

        setIsResizingCol(true);
        setResizingColIndex(colIndex);
        setResizeColStartX(e.clientX);
        setResizeColStartWidth(getColWidth(colIndex));
    };

    const handleColumnResizeMouseMove = useCallback((e) => {
        if (!isResizingCol || resizingColIndex === null) return;

        const delta = e.clientX - resizeColStartX;
        let newWidth = resizeColStartWidth + delta;

        // Apply min/max constraints
        const header = columnsHeaders[resizingColIndex];
        const minWidth = header?.minWidth ?? 30;
        const maxWidth = header?.maxWidth ?? 1000;
        newWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));

        const newHeaders = columnsHeaders.map((h, i) =>
            i === resizingColIndex ? { ...h, width: newWidth } : h
        );
        updateColumnsHeaders(newHeaders);
    }, [isResizingCol, resizingColIndex, resizeColStartX, resizeColStartWidth, columnsHeaders, updateColumnsHeaders]);

    const handleColumnResizeMouseUp = useCallback(() => {
        if (resizingColIndex !== null && onResizeColumn) {
            const finalWidth = getColWidth(resizingColIndex);
            onResizeColumn({ columnIndex: resizingColIndex, width: finalWidth });
        }
        setIsResizingCol(false);
        setResizingColIndex(null);
    }, [resizingColIndex, onResizeColumn, getColWidth]);

    // Row resize handlers
    const handleRowResizeMouseDown = (rowIndex, e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!canResizeRow(rowIndex)) return;

        setIsResizingRow(true);
        setResizingRowIndex(rowIndex);
        setResizeRowStartY(e.clientY);
        setResizeRowStartHeight(getRowHeight(rowIndex));
    };

    const handleRowResizeMouseMove = useCallback((e) => {
        if (!isResizingRow || resizingRowIndex === null) return;

        const delta = e.clientY - resizeRowStartY;
        let newHeight = resizeRowStartHeight + delta;

        // Apply min/max constraints
        const header = rowsHeaders[resizingRowIndex];
        const minHeight = header?.minHeight ?? 20;
        const maxHeight = header?.maxHeight ?? 200;
        newHeight = Math.max(minHeight, Math.min(maxHeight, newHeight));

        const newHeaders = rowsHeaders.map((h, i) =>
            i === resizingRowIndex ? { ...h, height: newHeight } : h
        );
        updateRowsHeaders(newHeaders);
    }, [isResizingRow, resizingRowIndex, resizeRowStartY, resizeRowStartHeight, rowsHeaders, updateRowsHeaders]);

    const handleRowResizeMouseUp = useCallback(() => {
        if (resizingRowIndex !== null && onResizeRow) {
            const finalHeight = getRowHeight(resizingRowIndex);
            onResizeRow({ rowIndex: resizingRowIndex, height: finalHeight });
        }
        setIsResizingRow(false);
        setResizingRowIndex(null);
    }, [resizingRowIndex, onResizeRow, getRowHeight]);

    // Effect for column/row resize
    useEffect(() => {
        if (isResizingCol) {
            document.addEventListener('mousemove', handleColumnResizeMouseMove);
            document.addEventListener('mouseup', handleColumnResizeMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleColumnResizeMouseMove);
                document.removeEventListener('mouseup', handleColumnResizeMouseUp);
            };
        }
    }, [isResizingCol, handleColumnResizeMouseMove, handleColumnResizeMouseUp]);

    useEffect(() => {
        if (isResizingRow) {
            document.addEventListener('mousemove', handleRowResizeMouseMove);
            document.addEventListener('mouseup', handleRowResizeMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleRowResizeMouseMove);
                document.removeEventListener('mouseup', handleRowResizeMouseUp);
            };
        }
    }, [isResizingRow, handleRowResizeMouseMove, handleRowResizeMouseUp]);

    // Apply fill operation
    const applyFill = () => {
        if (!selection.start || !fillDragEnd) return;

        const sourceValue = getCellValue(selection.start.row, selection.start.col);
        const sourceCell = cells[selection.start.row]?.[selection.start.col] || {};

        const minRow = Math.min(fillDragStart.row, fillDragEnd.row);
        const maxRow = Math.max(fillDragStart.row, fillDragEnd.row);
        const minCol = Math.min(fillDragStart.col, fillDragEnd.col);
        const maxCol = Math.max(fillDragStart.col, fillDragEnd.col);

        const newCells = cells.map((row, ri) =>
            row.map((cell, ci) => {
                if (ri >= minRow && ri <= maxRow && ci >= minCol && ci <= maxCol) {
                    if (isCellEditable(ri, ci)) {
                        return { ...cell, value: sourceValue };
                    }
                }
                return cell;
            })
        );

        updateCells(newCells);
    };

    // Handle fill handle mouse down
    const handleFillHandleMouseDown = (e) => {
        e.stopPropagation();
        if (!selection.start) return;
        setIsFillDragging(true);
        setFillDragStart(selection.start);
        setFillDragEnd(selection.end || selection.start);
    };

    // Handle edit input change
    const handleEditChange = (e) => {
        setEditValue(e.target.value);
    };

    // Handle edit input blur
    const handleEditBlur = () => {
        if (editingCell) {
            const newCells = cells.map((row, ri) =>
                row.map((cell, ci) => {
                    if (ri === editingCell.row && ci === editingCell.col) {
                        return { ...cell, value: editValue };
                    }
                    return cell;
                })
            );
            updateCells(newCells);
            setEditingCell(null);
        }
    };

    // Handle context menu - position at screen coordinates for portal
    const handleContextMenu = (e) => {
        e.preventDefault();

        // Use screen coordinates since we'll render via portal
        const menuWidth = 180;
        const menuHeight = 280;

        let x = e.clientX;
        let y = e.clientY;

        // Check if menu would go off right edge
        if (x + menuWidth > window.innerWidth) {
            x = e.clientX - menuWidth;
        }

        // Check if menu would go off bottom edge
        if (y + menuHeight > window.innerHeight) {
            y = e.clientY - menuHeight;
        }

        // Ensure menu doesn't go off top or left
        x = Math.max(0, x);
        y = Math.max(0, y);

        setContextMenu({
            visible: true,
            x: x,
            y: y
        });
    };

    // Close context menu
    const closeContextMenu = () => {
        setContextMenu({ visible: false, x: 0, y: 0 });
    };

    // Handle keyboard events
    const handleKeyDown = useCallback((e) => {
        if (!selection.start) return;

        // Ctrl+Z for undo
        if (e.ctrlKey && e.key === 'z') {
            e.preventDefault();
            undo();
            return;
        }

        // Ctrl+Y for redo
        if (e.ctrlKey && e.key === 'y') {
            e.preventDefault();
            redo();
            return;
        }

        // Ctrl+C for copy
        if (e.ctrlKey && e.key === 'c') {
            e.preventDefault();
            copySelection();
            return;
        }

        // Ctrl+V for paste
        if (e.ctrlKey && e.key === 'v') {
            e.preventDefault();
            pasteClipboard();
            return;
        }

        // Ctrl+X for cut
        if (e.ctrlKey && e.key === 'x') {
            e.preventDefault();
            cutSelection();
            return;
        }

        // Delete key
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (!editingCell) {
                e.preventDefault();
                deleteSelection();
            }
            return;
        }

        // Ctrl+Enter to fill selection with value
        if (e.ctrlKey && e.key === 'Enter') {
            if (editingCell) {
                e.preventDefault();
                fillSelectionWithValue(editValue);
                setEditingCell(null);
            }
            return;
        }

        // Enter to confirm edit or start editing
        if (e.key === 'Enter' && !e.ctrlKey) {
            if (editingCell) {
                handleEditBlur();
                // Move to next row
                const nextRow = Math.min(editingCell.row + 1, numRows - 1);
                updateSelection({ start: { row: nextRow, col: editingCell.col }, end: { row: nextRow, col: editingCell.col } });
            } else if (selection.start && isCellEditable(selection.start.row, selection.start.col)) {
                handleCellDoubleClick(selection.start.row, selection.start.col);
            }
            return;
        }

        // Escape to cancel edit or close context menu
        if (e.key === 'Escape') {
            setEditingCell(null);
            closeContextMenu();
            return;
        }

        // Arrow keys for navigation
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key) && !editingCell) {
            e.preventDefault();

            if (e.shiftKey) {
                // Extend selection from the current end point (or start if no end)
                const currentEnd = selection.end || selection.start;
                let newRow = currentEnd.row;
                let newCol = currentEnd.col;

                switch (e.key) {
                    case 'ArrowUp': newRow = Math.max(0, currentEnd.row - 1); break;
                    case 'ArrowDown': newRow = Math.min(numRows - 1, currentEnd.row + 1); break;
                    case 'ArrowLeft': newCol = Math.max(0, currentEnd.col - 1); break;
                    case 'ArrowRight': newCol = Math.min(numCols - 1, currentEnd.col + 1); break;
                }

                updateSelection({ start: selection.start, end: { row: newRow, col: newCol } });
            } else {
                // Move selection
                const { row, col } = selection.start;
                let newRow = row;
                let newCol = col;

                switch (e.key) {
                    case 'ArrowUp': newRow = Math.max(0, row - 1); break;
                    case 'ArrowDown': newRow = Math.min(numRows - 1, row + 1); break;
                    case 'ArrowLeft': newCol = Math.max(0, col - 1); break;
                    case 'ArrowRight': newCol = Math.min(numCols - 1, col + 1); break;
                }

                updateSelection({ start: { row: newRow, col: newCol }, end: { row: newRow, col: newCol } });
            }
        }

        // Tab for horizontal navigation
        if (e.key === 'Tab' && !editingCell) {
            e.preventDefault();
            const { row, col } = selection.start;
            const newCol = e.shiftKey ? Math.max(0, col - 1) : Math.min(numCols - 1, col + 1);
            updateSelection({ start: { row, col: newCol }, end: { row, col: newCol } });
        }

        // Start typing to edit
        if (!editingCell && !e.ctrlKey && !e.altKey && e.key.length === 1) {
            if (selection.start && isCellEditable(selection.start.row, selection.start.col)) {
                setEditingCell(selection.start);
                setEditValue(e.key);
            }
        }
    }, [selection, editingCell, editValue, numRows, numCols, cells, undo, redo]);

    // Copy selection
    const copySelection = () => {
        if (!selection.start) return;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);

        const copiedData = [];
        for (let r = minRow; r <= maxRow; r++) {
            const rowData = [];
            for (let c = minCol; c <= maxCol; c++) {
                rowData.push({ ...cells[r]?.[c] } || { value: '' });
            }
            copiedData.push(rowData);
        }
        setClipboard(copiedData);

        // Also copy to system clipboard as text
        const textData = copiedData.map(row => row.map(cell => cell.value ?? '').join('\t')).join('\n');
        navigator.clipboard?.writeText(textData);

        // Call onCopy event
        if (onCopy) {
            onCopy({ data: copiedData, bounds: { minRow, maxRow, minCol, maxCol } });
        }
    };

    // Cut selection
    const cutSelection = () => {
        if (!selection.start) return;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);

        copySelection();
        deleteSelection();

        // Call onCut event
        if (onCut) {
            onCut({ bounds: { minRow, maxRow, minCol, maxCol } });
        }
    };

    // Paste clipboard
    const pasteClipboard = async () => {
        if (!selection.start) return;

        let dataToPaste = clipboard;

        // Try to get from system clipboard
        try {
            const text = await navigator.clipboard?.readText();
            if (text) {
                dataToPaste = text.split('\n').map(row =>
                    row.split('\t').map(value => ({ value }))
                );
            }
        } catch (e) {
            // Use internal clipboard
        }

        if (!dataToPaste) return;

        const startRow = selection.start.row;
        const startCol = selection.start.col;

        const newCells = cells.map((row, ri) =>
            row.map((cell, ci) => {
                const pasteRow = ri - startRow;
                const pasteCol = ci - startCol;
                if (pasteRow >= 0 && pasteRow < dataToPaste.length &&
                    pasteCol >= 0 && pasteCol < dataToPaste[pasteRow].length) {
                    if (isCellEditable(ri, ci)) {
                        return { ...cell, value: dataToPaste[pasteRow][pasteCol].value };
                    }
                }
                return cell;
            })
        );

        updateCells(newCells);

        // Call onPaste event
        if (onPaste) {
            onPaste({ data: dataToPaste, startRow, startCol });
        }
    };

    // Delete selection
    const deleteSelection = () => {
        if (!selection.start) return;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);

        const newCells = cells.map((row, ri) =>
            row.map((cell, ci) => {
                if (ri >= minRow && ri <= maxRow && ci >= minCol && ci <= maxCol) {
                    if (isCellEditable(ri, ci)) {
                        return { ...cell, value: '' };
                    }
                }
                return cell;
            })
        );

        updateCells(newCells);
    };

    // Fill selection with value (Ctrl+Enter)
    const fillSelectionWithValue = (value) => {
        if (!selection.start) return;
        const minRow = Math.min(selection.start.row, selection.end?.row ?? selection.start.row);
        const maxRow = Math.max(selection.start.row, selection.end?.row ?? selection.start.row);
        const minCol = Math.min(selection.start.col, selection.end?.col ?? selection.start.col);
        const maxCol = Math.max(selection.start.col, selection.end?.col ?? selection.start.col);

        const newCells = cells.map((row, ri) =>
            row.map((cell, ci) => {
                if (ri >= minRow && ri <= maxRow && ci >= minCol && ci <= maxCol) {
                    if (isCellEditable(ri, ci)) {
                        return { ...cell, value };
                    }
                }
                return cell;
            })
        );

        updateCells(newCells);
    };

    // Add rows after the last selected row (or at end if no selection)
    const addRows = useCallback(() => {
        if (!canAddRows) return;
        const bounds = getSelectionBounds();
        
        // Determine number of columns for new rows
        const colCount = columnsHeaders ? columnsHeaders.length : (numCols || 1);
        
        // Handle empty table or no selection case
        if (!bounds) {
            // Add single row at the end
            if (maxRows && cells.length + 1 > maxRows) return;
            
            const newRow = Array(colCount).fill(null).map(() => ({ value: '' }));
            const newCells = [...cells, newRow];
            
            updateCells(newCells);
            closeContextMenu();
            
            // Select the newly added row
            const newRowIndex = newCells.length - 1;
            updateSelection({ start: { row: newRowIndex, col: 0 }, end: { row: newRowIndex, col: 0 } });
            
            if (onAddRow) {
                onAddRow({ addedRows: 1, insertAfterRow: cells.length - 1 });
            }
            return;
        }

        const numRowsToAdd = bounds.maxRow - bounds.minRow + 1;
        const insertAfterRow = bounds.maxRow;

        if (maxRows && cells.length + numRowsToAdd > maxRows) return;

        const newRows = Array(numRowsToAdd).fill(null).map(() =>
            Array(colCount).fill(null).map(() => ({ value: '' }))
        );

        const newCells = [
            ...cells.slice(0, insertAfterRow + 1),
            ...newRows,
            ...cells.slice(insertAfterRow + 1)
        ];

        updateCells(newCells);
        closeContextMenu();

        // Call onAddRow event
        if (onAddRow) {
            onAddRow({ addedRows: numRowsToAdd, insertAfterRow });
        }
    }, [canAddRows, selection, cells, numCols, columnsHeaders, maxRows, updateCells, onAddRow, updateSelection]);

    // Add columns after the last selected column
    const addColumns = useCallback(() => {
        if (!canAddColumns) return;
        const bounds = getSelectionBounds();
        if (!bounds) return;

        const numColsToAdd = bounds.maxCol - bounds.minCol + 1;
        const insertAfterCol = bounds.maxCol;

        if (maxColumns && numCols + numColsToAdd > maxColumns) return;

        const newCells = cells.map(row => {
            const newCellsForRow = Array(numColsToAdd).fill(null).map(() => ({ value: '' }));
            return [
                ...row.slice(0, insertAfterCol + 1),
                ...newCellsForRow,
                ...row.slice(insertAfterCol + 1)
            ];
        });

        updateCells(newCells);
        closeContextMenu();

        // Call onAddColumn event
        if (onAddColumn) {
            onAddColumn({ addedColumns: numColsToAdd, insertAfterCol });
        }
    }, [canAddColumns, selection, cells, numCols, maxColumns, updateCells, onAddColumn]);

    const deleteSelectedRows = useCallback(() => {
        if (!canDeleteRows) return;
        const bounds = getSelectionBounds();
        if (!bounds) return;

        const numRowsToDelete = bounds.maxRow - bounds.minRow + 1;

        // Check minRows constraint
        const minRowsLimit = minRows !== null ? minRows : 0;
        if (cells.length - numRowsToDelete < minRowsLimit) return;

        if (cells.length - numRowsToDelete < 1 && minRowsLimit < 1) return; // Always keep at least 1 row if minRows is 0/null? No, let's respect minRows explicitly or default logic.
        // Actually, if clear logic:
        if (cells.length - numRowsToDelete < minRowsLimit) return;

        const newCells = cells.filter((_, i) => i < bounds.minRow || i > bounds.maxRow);
        updateCells(newCells);
        updateSelection({ start: null, end: null });
        closeContextMenu();
        if (onDeleteRow) {
            onDeleteRow({ deletedRows: bounds.maxRow - bounds.minRow + 1, startRow: bounds.minRow });
        }
    }, [canDeleteRows, selection, cells, updateCells, onDeleteRow, updateSelection, minRows]);

    // Delete selected columns
    const deleteSelectedColumns = useCallback(() => {
        if (!canDeleteColumns) return;
        const bounds = getSelectionBounds();
        if (!bounds) return;

        const numColsToDelete = bounds.maxCol - bounds.minCol + 1;
        if (numCols - numColsToDelete < 1) return;

        const newCells = cells.map(row =>
            row.filter((_, i) => i < bounds.minCol || i > bounds.maxCol)
        );
        updateCells(newCells);
        updateSelection({ start: null, end: null });
        closeContextMenu();
        if (onDeleteColumn) {
            onDeleteColumn({ deletedColumns: bounds.maxCol - bounds.minCol + 1, startColumn: bounds.minCol });
        }
    }, [canDeleteColumns, selection, cells, numCols, updateCells, onDeleteColumn, updateSelection]);

    // Effect to focus input when editing
    useEffect(() => {
        if (editingCell && inputRef.current) {
            inputRef.current.focus();
            inputRef.current.select();
        }
    }, [editingCell]);

    // Effect for keyboard events
    useEffect(() => {
        const handleGlobalKeyDown = (e) => {
            if (tableRef.current?.contains(document.activeElement) ||
                tableRef.current === document.activeElement) {
                handleKeyDown(e);
            }
        };
        document.addEventListener('keydown', handleGlobalKeyDown);
        return () => document.removeEventListener('keydown', handleGlobalKeyDown);
    }, [handleKeyDown]);

    // Effect for mouse up
    useEffect(() => {
        document.addEventListener('mouseup', handleMouseUp);
        return () => document.removeEventListener('mouseup', handleMouseUp);
    }, [isFillDragging, fillDragStart, fillDragEnd]);

    // Effect to sync header scroll with cells scroll
    useEffect(() => {
        const scrollContainer = scrollContainerRef.current;
        if (!scrollContainer) return;

        const handleScroll = () => {
            if (headerScrollRef.current) {
                headerScrollRef.current.scrollLeft = scrollContainer.scrollLeft;
            }
        };

        scrollContainer.addEventListener('scroll', handleScroll);
        return () => scrollContainer.removeEventListener('scroll', handleScroll);
    }, []);

    // Calculate total dimensions
    const totalWidth = Array(numCols).fill(0).reduce((sum, _, i) => sum + getColWidth(i), 0);
    const totalHeight = Array(numRows).fill(0).reduce((sum, _, i) => sum + getRowHeight(i), 0);

    // Styles
    const containerStyle = {
        width: typeof width === 'number' ? `${width}px` : width,
        height: `${height}px`,
        border: '1px solid #ccc',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'Arial, sans-serif',
        fontSize: '14px',
        position: 'relative',
        outline: 'none',
        backgroundColor: '#fff',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none'
    };

    const headerRowStyle = {
        display: 'flex',
        flexDirection: 'row',
        flexShrink: 0,
        backgroundColor: '#f0f0f0',
        borderBottom: '2px solid #999'
    };

    const scrollContainerStyle = {
        flex: 1,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'row'
    };

    const rowHeadersContainerStyle = {
        flexShrink: 0,
        backgroundColor: '#f0f0f0',
        borderRight: '2px solid #999'
    };

    const cellsContainerStyle = {
        flex: 1,
        position: 'relative'
    };

    const cellStyle = (row, col) => {
        const style = getCellStyle(row, col);
        const isSelected = isCellSelected(row, col);
        const isEditing = editingCell?.row === row && editingCell?.col === col;
        const isInFillArea = isCellInFillArea(row, col);

        return {
            width: `${getColWidth(col)}px`,
            height: `${getRowHeight(row)}px`,
            minWidth: `${getColWidth(col)}px`,
            minHeight: `${getRowHeight(row)}px`,
            boxSizing: 'border-box',
            borderRight: '1px solid #ddd',
            borderBottom: '1px solid #ddd',
            padding: '4px 6px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            cursor: 'cell',
            position: 'relative',
            ...style,
            backgroundColor: isSelected ? '#cce5ff' : (isInFillArea ? '#e0e0ff' : style.backgroundColor),
            outline: isEditing ? '2px solid #0066cc' : (isSelected && selection.start?.row === row && selection.start?.col === col ? '2px solid #0066cc' : 'none'),
            outlineOffset: '-2px'
        };
    };

    const headerCellStyle = (header, isRow = false) => ({
        width: isRow ? `${headerColWidth}px` : `${header?.width || defaultColWidth}px`,
        height: isRow ? `${header?.height || defaultRowHeight}px` : `${headerRowHeight}px`,
        minWidth: isRow ? `${headerColWidth}px` : `${header?.width || defaultColWidth}px`,
        minHeight: isRow ? `${header?.height || defaultRowHeight}px` : `${headerRowHeight}px`,
        boxSizing: 'border-box',
        borderRight: '1px solid #999',
        borderBottom: isRow ? '1px solid #999' : 'none',
        padding: '4px 6px',
        fontWeight: header?.fontWeight || 'bold',
        textAlign: header?.align || 'center',
        backgroundColor: header?.backgroundColor || '#e0e0e0',
        color: header?.textColor || '#333',
        fontSize: header?.fontSize || '14px',
        fontFamily: header?.fontFamily || 'inherit',
        fontStyle: header?.fontStyle || 'normal',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        display: 'flex',
        alignItems: 'center',
        justifyContent: header?.align === 'right' ? 'flex-end' : (header?.align === 'center' ? 'center' : 'flex-start'),
        position: 'relative'
    });

    const fillHandleStyle = {
        position: 'absolute',
        right: '-10px',
        bottom: '-10px',
        width: '20px',
        height: '20px',
        backgroundColor: '#0066cc',
        cursor: 'crosshair',
        zIndex: 10,
        border: '3px solid #fff',
        boxSizing: 'border-box',
        borderRadius: '2px'
    };

    const columnResizeHandleStyle = {
        position: 'absolute',
        right: 0,
        top: 0,
        bottom: 0,
        width: '5px',
        cursor: 'col-resize',
        backgroundColor: 'transparent',
        zIndex: 5
    };

    const rowResizeHandleStyle = {
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 0,
        height: '5px',
        cursor: 'row-resize',
        backgroundColor: 'transparent',
        zIndex: 5
    };

    // Get bounds info for context menu labels
    const bounds = getSelectionBounds();
    const selectedColCount = bounds ? bounds.maxCol - bounds.minCol + 1 : 1;
    const selectedRowCount = bounds ? bounds.maxRow - bounds.minRow + 1 : 1;

    // Get selected cell position for fill handle
    const getSelectedCellPosition = () => {
        if (!selection.start) return null;
        const endRow = selection.end?.row ?? selection.start.row;
        const endCol = selection.end?.col ?? selection.start.col;
        const maxRow = Math.max(selection.start.row, endRow);
        const maxCol = Math.max(selection.start.col, endCol);

        let left = rowsHeaders ? headerColWidth : 0;
        for (let c = 0; c < maxCol; c++) {
            left += getColWidth(c);
        }
        left += getColWidth(maxCol);

        let top = 0;
        for (let r = 0; r < maxRow; r++) {
            top += getRowHeight(r);
        }
        top += getRowHeight(maxRow);

        return { left, top };
    };

    const fillHandlePosition = getSelectedCellPosition();

    return (
        <div
            ref={tableRef}
            style={containerStyle}
            tabIndex={0}
            onContextMenu={handleContextMenu}
        >
            {/* Column Headers */}
            {columnsHeaders && (
                <div style={{ display: 'flex', flexDirection: 'row', flexShrink: 0, backgroundColor: '#f0f0f0', borderBottom: '2px solid #999' }}>
                    {rowsHeaders && (
                        <div style={{
                            width: `${headerColWidth}px`,
                            height: `${headerRowHeight}px`,
                            backgroundColor: '#d0d0d0',
                            borderRight: '2px solid #999',
                            borderBottom: '2px solid #999',
                            flexShrink: 0
                        }} />
                    )}
                    <div
                        ref={headerScrollRef}
                        style={{
                            flex: 1,
                            overflow: 'hidden',
                            display: 'flex',
                            flexDirection: 'row'
                        }}
                    >
                        {columnsHeaders.map((header, i) => (
                            <div key={i} style={headerCellStyle(header)}>
                                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {header.name}
                                </span>
                                {canResizeColumn(i) && (
                                    <div
                                        style={columnResizeHandleStyle}
                                        onMouseDown={(e) => handleColumnResizeMouseDown(i, e)}
                                    />
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Scrollable area with row headers and cells */}
            <div ref={scrollContainerRef} style={scrollContainerStyle}>
                {/* Row Headers */}
                {rowsHeaders && (
                    <div style={rowHeadersContainerStyle}>
                        {rowsHeaders.map((header, i) => (
                            <div key={i} style={headerCellStyle(header, true)}>
                                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {header.name}
                                </span>
                                {canResizeRow(i) && (
                                    <div
                                        style={rowResizeHandleStyle}
                                        onMouseDown={(e) => handleRowResizeMouseDown(i, e)}
                                    />
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {/* Cells */}
                <div style={cellsContainerStyle}>
                    {cells.map((row, ri) => (
                        <div key={ri} style={{ display: 'flex' }}>
                            {row.map((cell, ci) => (
                                <div
                                    key={ci}
                                    style={cellStyle(ri, ci)}
                                    onClick={(e) => handleCellClick(ri, ci, e)}
                                    onDoubleClick={() => handleCellDoubleClick(ri, ci)}
                                    onMouseDown={(e) => handleMouseDown(ri, ci, e)}
                                    onMouseMove={() => handleMouseMove(ri, ci)}
                                >
                                    {editingCell?.row === ri && editingCell?.col === ci ? (
                                        <input
                                            ref={inputRef}
                                            type="text"
                                            value={editValue}
                                            onChange={handleEditChange}
                                            onBlur={handleEditBlur}
                                            style={{
                                                width: '100%',
                                                height: '100%',
                                                border: 'none',
                                                outline: 'none',
                                                padding: 0,
                                                margin: 0,
                                                font: 'inherit',
                                                backgroundColor: 'transparent',
                                                userSelect: 'text',
                                                WebkitUserSelect: 'text',
                                                MozUserSelect: 'text',
                                                msUserSelect: 'text'
                                            }}
                                        />
                                    ) : (
                                        cell.value ?? ''
                                    )}
                                    {/* Fill handle */}
                                    {selection.start?.row === ri && selection.start?.col === ci &&
                                        (!selection.end || (selection.end.row === ri && selection.end.col === ci)) && (
                                            <div
                                                style={fillHandleStyle}
                                                onMouseDown={handleFillHandleMouseDown}
                                            />
                                        )}
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            {/* Context Menu */}
            {contextMenu.visible && (
                <PopupMenu
                    position={{ x: contextMenu.x, y: contextMenu.y }}
                    onClose={closeContextMenu}
                    borderColor="#e0e0e0"
                    minWidth={190}
                    items={[
                        { label: 'Copy',  shortcut: 'Ctrl+C', onClick: copySelection },
                        { label: 'Cut',   shortcut: 'Ctrl+X', disabled: !isAnySelectedCellEditable(), onClick: cutSelection },
                        { label: 'Paste', shortcut: 'Ctrl+V', onClick: pasteClipboard },
                        { type: 'separator' },
                        { label: `Add Column${selectedColCount > 1 ? `s (${selectedColCount})` : ''}`,    disabled: !canAddColumns    || !!(maxColumns && numCols + selectedColCount > maxColumns),         onClick: addColumns },
                        { label: `Add Row${selectedRowCount > 1 ? `s (${selectedRowCount})` : ''}`,       disabled: !canAddRows       || !!(maxRows    && cells.length + selectedRowCount > maxRows),        onClick: addRows },
                        { label: `Delete Column${selectedColCount > 1 ? `s (${selectedColCount})` : ''}`, disabled: !canDeleteColumns || numCols - selectedColCount < 1,                                      onClick: deleteSelectedColumns },
                        { label: `Delete Row${selectedRowCount > 1 ? `s (${selectedRowCount})` : ''}`,    disabled: !canDeleteRows    || cells.length - selectedRowCount < 1,                                  onClick: deleteSelectedRows },
                    ]}
                />
            )}
        </div>
    );
};

export default Table;
