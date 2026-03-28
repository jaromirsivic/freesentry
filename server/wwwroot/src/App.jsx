import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './Layout';
import MainPage from './MainPage';
import HotZone from './HotZone';
import Settings from './Settings';
import Tools from './Tools';
import ManualControl from './ManualControl';
import Tutorials from './Tutorials';
import About from './About';
import AIAgent from './AIAgent';
import ImportExport from './ImportExport';
import GeneralSetup from './GeneralSetup';
import Motors from './Motors';
import AISetup from './AISetup';

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

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<MainPage />} />
          <Route path="manual-control" element={<ManualControl />} />
          <Route path="ai-agent" element={<AIAgent />} />
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
          <Route path="tutorials" element={<Tutorials />} />
          <Route path="sandbox" element={<Sandbox />}>
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
