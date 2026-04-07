import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './assets/color_palette.css'
import App from './App.jsx'
import { GeneralSettingsProvider } from './contexts/GeneralSettingsContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GeneralSettingsProvider>
      <App />
    </GeneralSettingsProvider>
  </StrictMode>,
)
