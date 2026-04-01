export const createOperationFeedback = (error, fallbackMessage) => ({
    tone: error?.ambiguous ? 'warning' : 'error',
    message: error?.userMessage || error?.message || fallbackMessage
});
