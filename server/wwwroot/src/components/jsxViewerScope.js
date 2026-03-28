import React from 'react';

function getComponentName(path) {
    return path.split('/').pop()?.replace(/\.jsx$/, '') ?? path;
}

const componentModules = import.meta.glob(['./*.jsx', '!./JSXViewer.jsx'], { eager: true });

export const jsxViewerSystemComponents = Object.freeze(
    Object.entries(componentModules)
        .sort(([left], [right]) => left.localeCompare(right))
        .reduce((components, [path, module]) => {
            if (!module?.default) {
                return components;
            }

            components[getComponentName(path)] = module.default;
            return components;
        }, {})
);

export const jsxViewerScope = Object.freeze({
    React,
    ...React,
    Fragment: React.Fragment,
    ...jsxViewerSystemComponents,
});

export default jsxViewerScope;
