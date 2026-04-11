import React, { useCallback } from 'react';
import MarkdownViewer from './components/MarkdownViewer';
import {
  getDocumentationPage,
  resolveDocumentationAssetUrl,
  resolveDocumentationMarkdownHref,
} from './documentationConfig';

const VIEWER_PADDING = 'clamp(1rem, 2vw, 2rem)';

const DocumentationPage = ({ pageId }) => {
  const page = getDocumentationPage(pageId);

  const handleResolveAssetUrl = useCallback(
    (target) => resolveDocumentationAssetUrl(target),
    [],
  );

  const handleResolveMarkdownHref = useCallback(
    (target) => resolveDocumentationMarkdownHref(target),
    [],
  );

  if (!page) {
    return (
      <div className="page-container" style={{ padding: '1rem' }}>
        <p>Documentation page not found.</p>
      </div>
    );
  }

  return (
    <div
      className="page-container"
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        height: '100%',
        backgroundColor: '#ffffff',
      }}
    >
      <MarkdownViewer
        value={page.document}
        docPath={page.docDirectory}
        resolveAssetUrl={handleResolveAssetUrl}
        resolveMarkdownHref={handleResolveMarkdownHref}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          backgroundColor: '#ffffff',
          padding: VIEWER_PADDING,
          paddingBottom: `calc(${VIEWER_PADDING} + var(--safe-area-bottom))`,
        }}
      />
    </div>
  );
};

export default DocumentationPage;
