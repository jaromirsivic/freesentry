import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './Layout';
import { useGeneralSettings } from './contexts/GeneralSettingsContext.jsx';
import MainPage from './MainPage';
import HotZone from './HotZone';
import Settings from './Settings';
import Tools from './Tools';
import ManualControl from './ManualControl';
import About from './About';
import AIAgent from './AIAgent';
import ImportExport from './ImportExport';
import GeneralSetup from './GeneralSetup';
import Motors from './Motors';
import AISetup from './AISetup';
import SystemGuide from './SystemGuide';
import WhatToBuy from './WhatToBuy';
import Electronics from './Electronics';

import Cameras from './Cameras';
import ComponentsDemo from './ComponentsDemo';
import Sandbox from './Sandbox';
import ModalWindowsDemo from './ModalWindowsDemo';
import EditableChartDemo from './EditableChartDemo';
import TableDemo from './TableDemo';
import Chart2DDemo from './Chart2DDemo';
import PolygonZoomPanDemo from './PolygonZoomPanDemo';
import DateTimePickerDemo from './DateTimePickerDemo';
import Joystick1DDemo from './Joystick1DDemo';

function DebugOnlyRouteGate({ children }) {
  const { isDebugMode, generalSettingsLoaded } = useGeneralSettings();
  if (!generalSettingsLoaded) {
    return null;
  }
  if (!isDebugMode) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function App() {
  const routerBasename =
    import.meta.env.BASE_URL === '/'
      ? '/'
      : import.meta.env.BASE_URL.replace(/\/$/, '');

  return (
    <BrowserRouter basename={routerBasename}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<MainPage />} />
          <Route path="manual-control" element={<ManualControl />} />
          <Route
            path="ai-agent"
            element={(
              <DebugOnlyRouteGate>
                <AIAgent />
              </DebugOnlyRouteGate>
            )}
          />
          <Route path="settings" element={<Settings />}>
            <Route path="import-export" element={<ImportExport />} />
            <Route path="general-setup" element={<GeneralSetup />} />
            <Route path="motors" element={<Motors />} />
            <Route path="cameras-new" element={<Cameras />} />
            <Route path="ai-behavior" element={<AISetup />} />
          </Route>
          <Route path="tools" element={<Tools />}>
            <Route path="hot-zone" element={<HotZone />} />
          </Route>
          <Route path="tutorials">
            <Route index element={<Navigate to="system-guide" replace />} />
            <Route path="system-guide" element={<SystemGuide />} />
            <Route path="what-to-buy" element={<WhatToBuy />} />
            <Route path="electronics" element={<Electronics />} />
          </Route>
          <Route
            path="sandbox"
            element={(
              <DebugOnlyRouteGate>
                <Sandbox />
              </DebugOnlyRouteGate>
            )}
          >
            <Route path="components-demo" element={<ComponentsDemo />} />
            <Route path="modal-windows-demo" element={<ModalWindowsDemo />} />
            <Route path="editable-chart" element={<EditableChartDemo />} />
            <Route path="table-demo" element={<TableDemo />} />
            <Route path="chart2d" element={<Chart2DDemo />} />
            <Route path="polygon-zoom-pan" element={<PolygonZoomPanDemo />} />
            <Route path="datetimepicker" element={<DateTimePickerDemo />} />
            <Route path="joystick1d" element={<Joystick1DDemo />} />
          </Route>
          <Route path="about" element={<About />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
