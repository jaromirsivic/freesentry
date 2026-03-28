from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
import subprocess
import time
from .context import master_controller
from .common import get_platform_info
from . import settingscontroller
from . import restapimotors
from . import restapicameras_old
from . import restapicameras
from . import restapisettings
from . import restapihotzone
from . import restapimanualcontrol
from . import restapiaisetup
from . import restapiosmanagement

async def execute_startup_script():
    """Load and execute the OS-specific startup script from settings.json."""
    os_code = get_platform_info().get("operating_system_code", "")
    print(f"Operating system code: {os_code}")
    try:
        settings = await settingscontroller.get_settings()
        startup_entry = settings.get("general", {}).get("startupScript", {}).get(os_code, None)
        if startup_entry is None:
            print(f"No startup script found for '{os_code}', skipping.")
            return
        language = startup_entry.get("language", "")
        script = startup_entry.get("script", "")
        if not script:
            print(f"Startup script for '{os_code}' is empty, skipping.")
            return
        print(f"Executing startup script for '{os_code}' (language: {language})...")
        if language == "python":
            exec(script)
        else:
            print(f"Unsupported startup script language: '{language}', skipping.")
    except Exception as e:
        print(f"Startup script error: {e}")

async def wifi_startup():
    """Configure wifi on Raspberry Pi based on settings.json."""
    os_code = get_platform_info().get("operating_system_code", "")
    if not os_code.startswith("raspberrypi"):
        return
    try:
        settings = await settingscontroller.get_settings()
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

async def onload():
    print("Server loaded")
    await wifi_startup()
    await execute_startup_script()
    print("Startup script execution completed")
    print("Open web browser at http://127.0.0.1 to access the application.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    print("Server starting up...")
    master_controller.start()            
    # onLoad function is called after the server is started
    await onload()
    yield
    # Shutdown logic goes here
    print("Server shutting down...")
    master_controller.stop()

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
app.include_router(restapisettings.router)
app.include_router(restapimotors.router)
app.include_router(restapihotzone.router)
app.include_router(restapicameras_old.router)
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
