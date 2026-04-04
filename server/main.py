from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
import subprocess
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from .context import (
    StartupState,
    clear_master_controller,
    clear_startup_state,
    get_startup_state_from_app,
    set_master_controller,
    set_startup_state,
)
from .mastercontroller import MasterController
from . import settingscontroller
from .platformcapabilities import HostCapabilities, get_host_capabilities
from . import restapihealth
from . import restapimotors
#from . import restapicameras_old
from . import restapicameras
from . import restapisettings
from . import restapihotzone
from . import restapimanualcontrol
from . import restapiaisetup
from . import restapiosmanagement

def _resolve_startup_script_entry(
    *,
    settings: Mapping[str, Any],
    capabilities: HostCapabilities,
) -> tuple[str | None, Mapping[str, Any] | None]:
    startup_scripts = settings.get("general", {}).get("startupScript", {})
    if not isinstance(startup_scripts, Mapping):
        return None, None

    for os_code in capabilities.startup_script_os_codes:
        startup_entry = startup_scripts.get(os_code)
        if isinstance(startup_entry, Mapping):
            return os_code, startup_entry

    return None, None


def _execute_startup_script_sync(*, settings: Mapping[str, Any], capabilities: HostCapabilities) -> None:
    """Load and execute the OS-specific startup script from settings.json."""
    print(f"Operating system code: {capabilities.operating_system_code}")
    try:
        selected_os_code, startup_entry = _resolve_startup_script_entry(
            settings=settings,
            capabilities=capabilities,
        )
        if startup_entry is None:
            print(f"No startup script found for '{capabilities.operating_system_code}', skipping.")
            return
        language = startup_entry.get("language", "")
        script = startup_entry.get("script", "")
        if not script:
            print(f"Startup script for '{selected_os_code}' is empty, skipping.")
            return
        print(f"Executing startup script for '{selected_os_code}' (language: {language})...")
        if language == "python":
            exec(script)
        else:
            print(f"Unsupported startup script language: '{language}', skipping.")
    except Exception as e:
        print(f"Startup script error: {e}")
        raise

def _wifi_startup_sync(*, settings: Mapping[str, Any], capabilities: HostCapabilities) -> None:
    """Configure wifi on Raspberry Pi based on settings.json."""
    if not capabilities.operating_system_code.startswith("raspberrypi"):
        return
    if not capabilities.supports_wifi_configuration:
        print("Wifi startup skipped: required Raspberry Pi wifi capabilities are unavailable.")
        return
    try:
        wifi = settings.get("general", {}).get("wifi", None)
        if wifi is None:
            print("No wifi configuration found in settings, skipping.")
            return
        mode = wifi.get("mode", "")
        print(f"Wifi mode: {mode}")
        if mode == "accessPoint":
            ap = wifi.get("accessPoint", {})
            ssid = ap.get("ssid", "")
            password = ap.get("password", "")
            print(f'Restarting wifi...')
            subprocess.run(["sudo", "nmcli", "radio", "wifi", "off"], check=True)
            time.sleep(1)
            subprocess.run(["sudo", "nmcli", "radio", "wifi", "on"], check=True)
            time.sleep(1)
            print(f'Starting wifi access point...')
            subprocess.run(["sudo", "nmcli", "device", "wifi", "hotspot",
                            "ifname", "wlan0", "ssid", ssid,
                            "password", password], check=True)
            print(f"Wifi access point '{ssid}' started.")
        elif mode == "client":
            client = wifi.get("client", {})
            connection_name = client.get("connectionName", "")
            ssid = client.get("ssid", "")
            security = client.get("security", None)
            password = client.get("password", "")
            subprocess.run(["sudo", "nmcli", "radio", "wifi", "off"], check=True)
            subprocess.run(["sudo", "nmcli", "radio", "wifi", "on"], check=True)
            base_cmd = ["sudo", "nmcli", "connection", "add", "type", "wifi",
                        "ifname", "wlan0", "con-name", connection_name,
                        "ssid", ssid]
            if security is None or security.lower() == "none":
                subprocess.run(base_cmd, check=True)
            elif security == "wep":
                subprocess.run(base_cmd + ["--",
                                           "wifi-sec.key-mgmt", "none",
                                           "wifi-sec.wep-key0", password,
                                           "wifi-sec.auth-alg", "open"], check=True)
            elif security == "wpa-psk":
                subprocess.run(base_cmd + ["--",
                                           "wifi-sec.key-mgmt", "wpa-psk",
                                           "wifi-sec.psk", password], check=True)
            elif security == "wpa3":
                subprocess.run(base_cmd + ["--",
                                           "wifi-sec.key-mgmt", "sae",
                                           "wifi-sec.psk", password], check=True)
            else:
                print(f"Unknown wifi security type: '{security}', skipping.")
                return
            print(f"Wifi client connection '{connection_name}' for '{ssid}' configured.")
        else:
            print(f"Unknown wifi mode: '{mode}', skipping.")
    except Exception as e:
        print(f"Wifi startup error: {e}")
        raise

def log_ai_runtime_diagnostics() -> None:
    """Emit one-shot runtime diagnostics for the active backend interpreter."""
    try:
        from .yolomodels import YOLOModels

        diagnostics = YOLOModels.get_runtime_diagnostics()
        print("AI runtime diagnostics:")
        print(f"  python: {diagnostics['python_executable']}")
        print(f"  torch: {diagnostics['torch_module_path']}")
        print(f"  torch_version: {diagnostics['torch_version']}")
        print(f"  torch_cuda_version: {diagnostics['torch_cuda_version']}")
        print(f"  cuda_available: {diagnostics['cuda_available']}")
        print(f"  cuda_device_count: {diagnostics['cuda_device_count']}")
        print(f"  default_device: {diagnostics['default_device']}")
    except Exception as e:
        print(f"AI runtime diagnostics failed: {e}")

async def _run_deferred_startup(app: FastAPI) -> None:
    startup_state = get_startup_state_from_app(app)

    try:
        capabilities = get_host_capabilities()
        await asyncio.to_thread(log_ai_runtime_diagnostics)
        settings = await settingscontroller.get_settings()
        await asyncio.to_thread(_wifi_startup_sync, settings=settings, capabilities=capabilities)
        await asyncio.to_thread(
            _execute_startup_script_sync,
            settings=settings,
            capabilities=capabilities,
        )
    except asyncio.CancelledError:
        print("Deferred startup task cancelled.")
        raise
    except Exception as e:
        startup_state.mark_failed(e)
        print(f"Deferred startup error: {e}")
    else:
        startup_state.mark_ready()
        print("Startup script execution completed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    master_controller = MasterController()
    startup_state = StartupState()
    set_master_controller(app, master_controller)
    set_startup_state(app, startup_state)
    try:
        master_controller.start()
        print("Server loaded")
        startup_state.mark_post_start()
        startup_state.deferred_startup_task = asyncio.create_task(
            _run_deferred_startup(app),
            name="deferred-startup",
        )
        print("Open web browser at http://127.0.0.1 to access the application.")
        yield
    finally:
        print("Server shutting down...")
        try:
            startup_state = get_startup_state_from_app(app)
        except RuntimeError:
            startup_state = None
        if startup_state is not None:
            startup_state.mark_stopping()
            deferred_startup_task = startup_state.deferred_startup_task
            if deferred_startup_task is not None and not deferred_startup_task.done():
                deferred_startup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await deferred_startup_task
        try:
            master_controller.stop()
        finally:
            clear_startup_state(app)
            clear_master_controller(app)

app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include Routers
app.include_router(restapihealth.router)
app.include_router(restapisettings.router)
app.include_router(restapimotors.router)
app.include_router(restapihotzone.router)
#app.include_router(restapicameras_old.router)
app.include_router(restapicameras.router)
app.include_router(restapimanualcontrol.router)
app.include_router(restapiaisetup.router)
app.include_router(restapiosmanagement.router)


@app.exception_handler(404)
async def spa_fallback_handler(request: Request, exc: StarletteHTTPException):
    """
    Fallback handler for Single Page Application routing.
    If a 404 occurs and the path is NOT an API endpoint, serve index.html.
    This allows React Router to handle the routing client-side.
    """
    if request.url.path.startswith("/api") or request.url.path.startswith("/ws"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    
    # Serve index.html for SPA routes
    index_path = BASE_DIR / "wwwroot/dist/index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent

app.mount("/", StaticFiles(directory=BASE_DIR / "wwwroot/dist", html=True), name="static")
