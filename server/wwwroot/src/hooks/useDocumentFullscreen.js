import { useCallback, useEffect, useRef, useState } from 'react';

export default function useDocumentFullscreen() {
    const [isFullscreen, setIsFullscreen] = useState(false);
    const isFullscreenRef = useRef(false);

    useEffect(() => {
        isFullscreenRef.current = isFullscreen;
    }, [isFullscreen]);

    useEffect(() => {
        const handleFullscreenChange = () => {
            const isNowFullscreen = !!(
                document.fullscreenElement
                || document.webkitFullscreenElement
                || document.mozFullScreenElement
                || document.msFullscreenElement
            );

            isFullscreenRef.current = isNowFullscreen;
            setIsFullscreen(isNowFullscreen);

            if (isNowFullscreen) {
                document.body.classList.add('is-fullscreen');
            } else {
                document.body.classList.remove('is-fullscreen');
            }
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('mozfullscreenchange', handleFullscreenChange);
        document.addEventListener('MSFullscreenChange', handleFullscreenChange);

        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
            document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
            document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
            document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
            document.body.classList.remove('is-fullscreen');
        };
    }, []);

    const toggleFullscreen = useCallback(async () => {
        try {
            if (!isFullscreenRef.current) {
                const element = document.documentElement;
                if (element.requestFullscreen) {
                    await element.requestFullscreen();
                } else if (element.webkitRequestFullscreen) {
                    await element.webkitRequestFullscreen();
                } else if (element.mozRequestFullScreen) {
                    await element.mozRequestFullScreen();
                } else if (element.msRequestFullscreen) {
                    await element.msRequestFullscreen();
                }
            } else if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                await document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
                await document.msExitFullscreen();
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    }, []);

    return {
        isFullscreen,
        isFullscreenRef,
        toggleFullscreen
    };
}
