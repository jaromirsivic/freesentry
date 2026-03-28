import React, { useRef } from 'react';
import Button from './Button';
import { getOutlineStyle } from '../lib/outlineStyle';

const FileInput = ({
    label,
    onFileSelect,
    accept,
    disabled = false,
    labelWidth,
    style = {},
    outline = null
}) => {
    const fileInputRef = useRef(null);

    const handleButtonClick = () => {
        if (!disabled && fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file && onFileSelect) {
            onFileSelect(file);
        }
        // Reset value to allow selecting the same file again
        e.target.value = '';
    };

    return (
        <div className="responsive-input-container" style={{ width: '100%', opacity: disabled ? 0.5 : 1, gap: '1rem', ...getOutlineStyle(outline), ...style }}>
            {label && <span style={{ whiteSpace: 'nowrap', width: labelWidth, minWidth: labelWidth, display: labelWidth ? 'inline-block' : 'inline' }}>{label}</span>}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept={accept}
                style={{ display: 'none' }}
                disabled={disabled}
            />
            <Button
                label="Choose File..."
                onClick={handleButtonClick}
                disabled={disabled}
                style={{ width: 'auto' }}
            />
        </div>
    );
};

export default FileInput;
