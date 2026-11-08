"""
Suprema BioMini Slim 2 Scanner Integration
==========================================

This module provides integration with Suprema BioMini Slim 2 fingerprint scanner.

Requirements:
1. Install Suprema SDK from: https://www.supremainc.com/en/support/download-center.asp
2. Install the required Python package: pip install pywin32

The Suprema SDK typically includes:
- UFScanner.dll (or UFS_API.dll)
- Python bindings or COM interface

Note: This is a template implementation. You need to:
1. Download and install the Suprema BioMini SDK
2. Update the DLL path to match your installation
3. Install any required Python packages (usually pywin32 for Windows COM)
"""

import os
import sys
import time
from datetime import datetime
from PIL import Image
import numpy as np
from io import BytesIO
import subprocess
import ctypes
from ctypes import c_int, c_uint8, c_void_p, byref, create_string_buffer

def check_scanner_device_connected():
    """Check if Suprema BioMini scanner is connected via Windows Device Manager"""
    try:
        cmd = 'Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -like "*Suprema*" -or $_.FriendlyName -like "*BioMini*" -or $_.FriendlyName -like "*Authentication Scanner*" -or $_.Manufacturer -like "*Suprema*" } | Select-Object -First 1'
        result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=5)
        return len(result.stdout.strip()) > 0
    except Exception as e:
        print(f"Device check error: {e}")
        return False

class BioMiniScanner:
    def __init__(self):
        # Overall availability (direct or bridge)
        self.scanner_available = False
        # COM interface (optional)
        self.scanner = None
        # DLL interface (preferred)
        self.dll = None
        self._dll_path = None
        self._dll_dir = None
        self._using_dll = False
        self._hScanner = c_void_p(None)
        self._bound = False
        # Bridge mode
        self.use_demo_bridge = False
        self.initialize_scanner()
    
    def initialize_scanner(self):
        """Initialize connection to Suprema BioMini Slim 2 scanner"""
        try:
            # Get project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Method 1: Try using ctypes to load DLL directly (preferred for local SDK)
            try:
                import ctypes
                import platform
                
                # Determine architecture
                is_64bit = platform.machine().endswith('64')

                # Build search paths
                if is_64bit:
                    possible_paths = [
                        os.path.join(project_dir, r"Suprema\BioMini\bin\x64\UFScanner.dll"),
                        r"C:\\Program Files\\Suprema\\BioMini\\x64\\UFScanner.dll",
                        os.path.join(project_dir, r"Suprema\BioMini\bin\UFScanner.dll"),
                    ]
                else:
                    possible_paths = [
                        os.path.join(project_dir, r"Suprema\BioMini\bin\UFScanner.dll"),
                        r"C:\\Program Files (x86)\\Suprema\\BioMini\\UFScanner.dll",
                    ]

                dll_path = None
                dll_dir = None
                for pth in possible_paths:
                    if os.path.exists(pth):
                        dll_path = pth
                        dll_dir = os.path.dirname(pth)
                        break

                if dll_path:
                    # Add DLL dir so dependencies resolve
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(dll_dir)
                    os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

                    # Load stdcall DLL
                    try:
                        self.dll = ctypes.WinDLL(dll_path)
                    except Exception:
                        self.dll = ctypes.CDLL(dll_path)

                    # Bind and open first scanner
                    self._bind_dll_functions()
                    if self._dll_init_and_open_first_scanner():
                        self.scanner_available = True
                        self._using_dll = True
                        self.use_demo_bridge = False
                        self._dll_path = dll_path
                        self._dll_dir = dll_dir
                        print(f"✓ Scanner DLL loaded from: {dll_path}")
                        print(f"  Architecture: {'64-bit' if is_64bit else '32-bit'}")
                        print(f"  DLL directory: {dll_dir}")
                        return True
                    else:
                        # Init/open failed, continue to other methods
                        self.dll = None
                else:
                    print("⚠ Scanner DLL not found in expected locations")
                    print("  Searched paths:")
                    for p in possible_paths:
                        print(f"    - {p} {'[EXISTS]' if os.path.exists(p) else '[NOT FOUND]'}")
                    
            except Exception as e:
                print(f"DLL loading failed: {e}")
                # Don't print full traceback anymore
            
            # Method 2: Try using pywin32 COM interface (requires admin registration)
            try:
                import win32com.client
                self.scanner = win32com.client.Dispatch("UFS.UFS_API")
                self.scanner_available = True
                print("✓ Scanner initialized via COM interface")
                return True
            except Exception as e:
                print(f"COM interface not available: {e}")
            
            # Method 3: Check if device is physically connected
            # Even if DLL/COM fails, we can use demo bridge
            print("\n⚠ Direct DLL/COM access failed.")
            print("Checking if scanner device is connected...")
            
            if check_scanner_device_connected():
                print("✓ Scanner device detected! Enabling demo bridge mode...")
                self.scanner_available = True
                self.use_demo_bridge = True
                return True
            else:
                print("✗ No scanner device found.")
                print("\nTo enable scanner:")
                print("1. Install device drivers from: Suprema HQ Inc\\Suprema_Fingerprint_Scanner_Driver\\")
                print("2. Connect BioMini Slim 2 via USB")
                print("3. Restart this script")
                self.scanner_available = False
                return False
            
        except Exception as e:
            print(f"Scanner initialization error: {e}")
            self.scanner_available = False
            return False
    
    def is_available(self):
        """Check if scanner is available"""
        return self.scanner_available
    
    def capture_fingerprint(self):
        """
        Capture fingerprint from scanner and return as PIL Image
        
        Returns:
            PIL.Image: Captured fingerprint image
            None: If capture fails
        """
        if not self.scanner_available:
            print("Scanner not available - using simulation mode")
            return self._simulate_capture()
        
        try:
            # Demo bridge mode: launch sample app and wait for a saved image
            if self.use_demo_bridge:
                print("Using demo bridge mode: opening Suprema demo to capture image...")
                img = self._capture_via_demo_bridge(timeout_seconds=120)
                if img is not None:
                    print("✓ Fingerprint captured via demo bridge")
                    return img
                print("✗ Demo bridge did not produce an image within timeout")
                return None

            # Real scanner capture logic via COM interface (if available)
            if hasattr(self.scanner, 'CaptureImage'):
                result = self.scanner.CaptureImage()
                if result:
                    # Convert scanner image to PIL Image
                    # The format depends on SDK (usually BMP or raw bitmap)
                    image_data = self.scanner.GetImageData()
                    width = self.scanner.GetImageWidth()
                    height = self.scanner.GetImageHeight()
                    
                    # Convert to numpy array and then PIL Image
                    img_array = np.frombuffer(image_data, dtype=np.uint8)
                    img_array = img_array.reshape((height, width))
                    img = Image.fromarray(img_array, mode='L')
                    
                    print("✓ Fingerprint captured successfully")
                    return img

            # DLL interface (preferred)
            if self._using_dll and self.dll and self._hScanner:
                return self._capture_via_dll()

            # If no direct method available, fallback to demo bridge
            print("⚠ Direct capture not available; using demo bridge fallback")
            img = self._capture_via_demo_bridge(timeout_seconds=120)
            if img is not None:
                print("✓ Fingerprint captured via demo bridge (fallback)")
                return img
            return None
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None

    # -------------------------
    # DLL BINDINGS AND CAPTURE
    # -------------------------
    def _bind_dll_functions(self):
        if not self.dll or self._bound:
            return
        # Status type is int
        c_status = c_int
        # UFS_Init / Uninit
        self.dll.UFS_Init.restype = c_status
        self.dll.UFS_Uninit.restype = c_status
        # Update (optional)
        self.dll.UFS_Update.restype = c_status
        # Scanner enumeration
        self.dll.UFS_GetScannerNumber.argtypes = [ctypes.POINTER(c_int)]
        self.dll.UFS_GetScannerNumber.restype = c_status
        self.dll.UFS_GetScannerHandle.argtypes = [c_int, ctypes.POINTER(c_void_p)]
        self.dll.UFS_GetScannerHandle.restype = c_status
        # Parameters
        self.dll.UFS_SetParameter.argtypes = [c_void_p, c_int, ctypes.c_void_p]
        self.dll.UFS_SetParameter.restype = c_status
        # Capture
        self.dll.UFS_CaptureSingleImage.argtypes = [c_void_p]
        self.dll.UFS_CaptureSingleImage.restype = c_status
        # Buffer info
        self.dll.UFS_GetCaptureImageBufferInfo.argtypes = [c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int), ctypes.POINTER(c_int)]
        self.dll.UFS_GetCaptureImageBufferInfo.restype = c_status
        self.dll.UFS_GetCaptureImageBuffer.argtypes = [c_void_p, ctypes.POINTER(c_uint8)]
        self.dll.UFS_GetCaptureImageBuffer.restype = c_status
        self.dll.UFS_ClearCaptureImageBuffer.argtypes = [c_void_p]
        self.dll.UFS_ClearCaptureImageBuffer.restype = c_status
        # Optional error string helper
        try:
            self.dll.UFS_GetErrorString.argtypes = [c_int, ctypes.c_char_p]
            self.dll.UFS_GetErrorString.restype = c_status
        except Exception:
            pass
        self._bound = True

    def _errstr(self, code: int) -> str:
        try:
            buf = create_string_buffer(256)
            res = self.dll.UFS_GetErrorString(c_int(code), buf)
            if res == 0:
                return buf.value.decode(errors='ignore')
        except Exception:
            pass
        return f"UFS error {code}"

    def _dll_init_and_open_first_scanner(self) -> bool:
        try:
            # Initialize library
            res = self.dll.UFS_Init()
            if res != 0:
                print(f"UFS_Init failed: {self._errstr(res)}")
                return False
            # Update and enumerate
            try:
                self.dll.UFS_Update()
            except Exception:
                pass
            n = c_int(0)
            res = self.dll.UFS_GetScannerNumber(byref(n))
            if res != 0:
                print(f"UFS_GetScannerNumber failed: {self._errstr(res)}")
                return False
            if n.value <= 0:
                print("No scanners found by DLL")
                return False
            # Open first scanner (index 0)
            h = c_void_p(None)
            res = self.dll.UFS_GetScannerHandle(c_int(0), byref(h))
            if res != 0 or not h:
                print(f"UFS_GetScannerHandle failed: {self._errstr(res)}")
                return False
            self._hScanner = h
            # Set capture timeout (milliseconds)
            UFS_PARAM_TIMEOUT = 201
            timeout_ms = c_int(15000)  # 15s
            res = self.dll.UFS_SetParameter(self._hScanner, c_int(UFS_PARAM_TIMEOUT), ctypes.cast(byref(timeout_ms), ctypes.c_void_p))
            if res != 0:
                print(f"UFS_SetParameter(TIMEOUT) failed: {self._errstr(res)}")
            return True
        except Exception as e:
            print(f"DLL init/open error: {e}")
            return False

    def _capture_via_dll(self):
        """Capture a single image via UFScanner.dll and return PIL Image or None."""
        if not (self.dll and self._hScanner):
            return None
        # Clear any previous buffer
        try:
            self.dll.UFS_ClearCaptureImageBuffer(self._hScanner)
        except Exception:
            pass
        # Capture
        res = self.dll.UFS_CaptureSingleImage(self._hScanner)
        if res != 0:
            print(f"UFS_CaptureSingleImage failed: {self._errstr(res)}")
            return None
        # Get buffer info
        w = c_int(0); h = c_int(0); dpi = c_int(0)
        res = self.dll.UFS_GetCaptureImageBufferInfo(self._hScanner, byref(w), byref(h), byref(dpi))
        if res != 0 or w.value <= 0 or h.value <= 0:
            print(f"BufferInfo failed: {self._errstr(res)}")
            return None
        size = w.value * h.value
        buf = (c_uint8 * size)()
        res = self.dll.UFS_GetCaptureImageBuffer(self._hScanner, buf)
        if res != 0:
            print(f"GetCaptureImageBuffer failed: {self._errstr(res)}")
            return None
        # Convert to PIL Image (grayscale L)
        try:
            np_arr = np.frombuffer(buf, dtype=np.uint8).reshape((h.value, w.value))
            img = Image.fromarray(np_arr, mode='L')
            return img
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None

    def _candidate_demo_apps(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        demo_dir = os.path.join(project_dir, r"Suprema\BioMini\bin")
        return [
            os.path.join(demo_dir, "Image_DemoVC.exe"),
            os.path.join(demo_dir, "Image_DemoCS.exe"),
            os.path.join(demo_dir, "MultiScannerDemo.exe"),
            os.path.join(demo_dir, "DiagnosisTool.exe"),
        ]

    def _watch_dirs_for_new_image(self, start_time: float, timeout_seconds: int, preferred_dir: str | None = None):
        """Poll known folders for newly created image files after start_time."""
        project_dir = os.path.dirname(os.path.abspath(__file__))
        bin_dir = os.path.join(project_dir, r"Suprema\BioMini\bin")
        x64_dir = os.path.join(bin_dir, "x64")
        user_home = os.path.expanduser("~")
        pictures = os.path.join(user_home, "Pictures")
        desktop = os.path.join(user_home, "Desktop")
        documents = os.path.join(user_home, "Documents")
        watch_dirs = []
        if preferred_dir and os.path.isdir(preferred_dir):
            watch_dirs.append(preferred_dir)
        for d in [bin_dir, x64_dir, project_dir, pictures, desktop, documents]:
            if os.path.isdir(d) and d not in watch_dirs:
                watch_dirs.append(d)
        exts = {".bmp", ".png", ".jpg", ".jpeg"}

        deadline = time.time() + timeout_seconds
        newest_path = None
        newest_mtime = start_time

        while time.time() < deadline:
            for d in watch_dirs:
                try:
                    for name in os.listdir(d):
                        _, ext = os.path.splitext(name.lower())
                        if ext in exts:
                            p = os.path.join(d, name)
                            try:
                                mtime = os.path.getmtime(p)
                            except OSError:
                                continue
                            if mtime > newest_mtime and mtime > start_time:
                                newest_mtime = mtime
                                newest_path = p
                except Exception:
                    pass
            if newest_path:
                # small settle delay to ensure file is fully written
                time.sleep(0.3)
                return newest_path
            time.sleep(0.8)
        return None

    def _capture_via_demo_bridge(self, timeout_seconds: int = 120):
        """
        Launch a Suprema demo app and wait until it saves an image in a known folder.
        This requires the user to press the Capture button in the demo window.
        """
        # Record start time
        start_time = time.time()
        demo_started = False
        last_error = None
        for exe in self._candidate_demo_apps():
            if os.path.exists(exe):
                try:
                    print(f"Launching demo app: {exe}")
                    # Launch without waiting; allow user to click Capture
                    subprocess.Popen([exe], cwd=os.path.dirname(exe))
                    demo_started = True
                    break
                except Exception as e:
                    last_error = e
                    print(f"Failed to launch {exe}: {e}")

        if not demo_started:
            print("No demo application could be launched.")
            if last_error:
                print(f"Last error: {last_error}")
            return None

        # Prepare a temp folder for saving
        project_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(project_dir, "temp_scans")
        os.makedirs(temp_dir, exist_ok=True)

        # Open the folder so user can save into it easily
        try:
            subprocess.Popen(["explorer", temp_dir])
        except Exception:
            pass

        print("A demo window should be open. Please place finger on the scanner and click 'Capture' in the demo.")
        print("Then click 'Save' in the demo and save the image into this folder:")
        print(f"  {temp_dir}")
        print("Waiting for the image to appear... (up to %ds)" % timeout_seconds)

        image_path = self._watch_dirs_for_new_image(start_time=start_time, timeout_seconds=timeout_seconds, preferred_dir=temp_dir)
        if image_path:
            try:
                img = Image.open(image_path).convert("L")
                print(f"Found image: {image_path}")
                return img
            except Exception as e:
                print(f"Failed to open captured image {image_path}: {e}")
                return None
        return None
    
    def _simulate_capture(self):
        """
        Simulation mode: Returns None to indicate user should upload manually
        In production, you could return a test fingerprint image
        """
        print("Simulation mode: Please upload fingerprint manually")
        return None
    
    def close(self):
        """Close scanner connection"""
        try:
            if self._using_dll and self.dll:
                try:
                    self.dll.UFS_Uninit()
                except Exception:
                    pass
                print("Scanner DLL uninitialized")
            elif self.scanner_available and self.scanner:
                # Call appropriate cleanup method based on SDK
                if hasattr(self.scanner, 'Close'):
                    self.scanner.Close()
                print("Scanner COM connection closed")
        except Exception as e:
            print(f"Error closing scanner: {e}")

    # Debug info for status endpoint/UI
    def debug_info(self):
        try:
            return {
                "available": bool(self.scanner_available),
                "using_dll": bool(self._using_dll),
                "use_demo_bridge": bool(self.use_demo_bridge),
                "dll_path": getattr(self, "_dll_path", None),
                "dll_dir": getattr(self, "_dll_dir", None),
            }
        except Exception:
            return {"available": False}


# Global scanner instance
_scanner_instance = None

def get_scanner():
    """Get or create global scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = BioMiniScanner()
    return _scanner_instance

def capture_fingerprint_image():
    """
    Convenience function to capture fingerprint
    Returns PIL Image or None
    """
    scanner = get_scanner()
    return scanner.capture_fingerprint()

def is_scanner_available():
    """Check if scanner is available"""
    scanner = get_scanner()
    if scanner.is_available():
        return True
    # Fallback: if device is connected, enable bridge mode dynamically
    try:
        if check_scanner_device_connected():
            scanner.scanner_available = True
            scanner.use_demo_bridge = True
            return True
    except Exception:
        pass
    return False

def get_scanner_debug():
    """Return internal debug info for diagnostics in status endpoint."""
    try:
        return get_scanner().debug_info()
    except Exception:
        return {"available": False}

# -------- Demo bridge helper functions exposed for API layer --------
def get_temp_scans_dir() -> str:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp_scans")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def open_demo_bridge_only() -> dict:
    """Launch demo app and open the temp_scans folder without waiting."""
    temp_dir = get_temp_scans_dir()
    # Try launching one of the demos
    demo_started = False
    last_error = None
    for exe in [
        os.path.join(os.path.dirname(temp_dir), r"Suprema\BioMini\bin\Image_DemoCS.exe"),
        os.path.join(os.path.dirname(temp_dir), r"Suprema\BioMini\bin\MultiScannerDemo.exe"),
    ]:
        if os.path.exists(exe):
            try:
                subprocess.Popen([exe], cwd=os.path.dirname(exe))
                demo_started = True
                break
            except Exception as e:
                last_error = e
                continue
    # Open folder for convenience
    try:
        subprocess.Popen(["explorer", temp_dir])
    except Exception:
        pass
    return {
        "started": demo_started,
        "save_dir": temp_dir,
        "error": str(last_error) if (not demo_started and last_error) else None
    }

def consume_latest_image(max_age_seconds: int = 300):
    """Return the latest image file from temp_scans if present and recent; else None."""
    temp_dir = get_temp_scans_dir()
    if not os.path.isdir(temp_dir):
        return None
    exts = {".bmp", ".png", ".jpg", ".jpeg"}
    latest = None
    latest_mtime = 0.0
    now = time.time()
    for name in os.listdir(temp_dir):
        fp = os.path.join(temp_dir, name)
        try:
            if not os.path.isfile(fp):
                continue
            _, ext = os.path.splitext(name.lower())
            if ext not in exts:
                continue
            m = os.path.getmtime(fp)
            if now - m <= max_age_seconds and m > latest_mtime:
                latest_mtime = m
                latest = fp
        except Exception:
            continue
    if not latest:
        return None
    try:
        img = Image.open(latest).convert("L")
        return {"path": latest, "image": img}
    except Exception:
        return None


if __name__ == "__main__":
    # Test scanner connection
    print("Testing Suprema BioMini Slim 2 Scanner...")
    print("=" * 50)
    
    scanner = get_scanner()
    
    print(f"\nScanner available: {scanner.is_available()}")
    
    if scanner.is_available():
        print("\nAttempting to capture fingerprint...")
        img = scanner.capture_fingerprint()
        if img:
            print(f"Success! Image size: {img.size}")
            # Save test image
            img.save("test_capture.png")
            print("Saved to test_capture.png")
    else:
        print("\nPlease install Suprema BioMini SDK:")
        print("1. Download from: https://www.supremainc.com/en/support/download-center.asp")
        print("2. Install the SDK for BioMini Slim 2")
        print("3. Install pywin32: pip install pywin32")
        print("4. Restart this script")
    
    scanner.close()
