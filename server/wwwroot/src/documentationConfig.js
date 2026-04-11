const DOCUMENT_SOURCE_CONTENT = import.meta.glob('../documentation/**/document.md', {
  eager: true,
  import: 'default',
  query: '?raw',
});

const DOCUMENT_ASSET_URLS = import.meta.glob('../documentation/**/*', {
  eager: true,
  import: 'default',
  query: '?url',
});

function buildDocumentationAssetKey(path = '', name = '') {
  return `../documentation/${path ? `${path}/` : ''}${name}`;
}

function buildDocumentationFileKey(path = '', name = '') {
  return `${path ? `${path}/` : ''}${name}`;
}

export const DOCUMENTATION_PAGES = Object.freeze({
  systemGuide: Object.freeze({
    title: 'System Guide',
    route: '/tutorials/system-guide',
    docDirectory: 'systemguide',
    document: DOCUMENT_SOURCE_CONTENT['../documentation/systemguide/document.md'] ?? '',
  }),
  whatToBuy: Object.freeze({
    title: 'What to Buy',
    route: '/tutorials/what-to-buy',
    docDirectory: 'whattobuy',
    document: DOCUMENT_SOURCE_CONTENT['../documentation/whattobuy/document.md'] ?? '',
  }),
  electronics: Object.freeze({
    title: 'Electronics',
    route: '/tutorials/electronics',
    docDirectory: 'electronics',
    document: DOCUMENT_SOURCE_CONTENT['../documentation/electronics/document.md'] ?? '',
  }),
});

const DOCUMENT_ROUTE_BY_FILE = new Map(
  Object.values(DOCUMENTATION_PAGES).map((page) => [
    buildDocumentationFileKey(page.docDirectory, 'document.md'),
    page.route,
  ]),
);

export function getDocumentationPage(pageId) {
  return DOCUMENTATION_PAGES[pageId] ?? null;
}

export function resolveDocumentationAssetUrl(target) {
  if (!target?.name) {
    return null;
  }

  return DOCUMENT_ASSET_URLS[buildDocumentationAssetKey(target.path, target.name)] ?? null;
}

export function resolveDocumentationMarkdownHref(target) {
  if (!target?.name) {
    return null;
  }

  return DOCUMENT_ROUTE_BY_FILE.get(buildDocumentationFileKey(target.path, target.name)) ?? null;
}
