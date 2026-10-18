#!/usr/bin/env python3
"""Exercise the real desktop renderer through X11, without Pillow or xdotool."""
import argparse
import ctypes as C
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=['SFML', 'QT', 'GTK'], required=True)
parser.add_argument('--executable', required=True)
args = parser.parse_args()
X = C.CDLL('libX11.so.6')
T = C.CDLL('libXtst.so.6')
Window = C.c_ulong
Display = C.c_void_p

def function(lib, name, result, *arguments):
    f = getattr(lib, name)
    f.restype, f.argtypes = result, list(arguments)
    return f

open_display = function(X, 'XOpenDisplay', Display, C.c_char_p)
display = open_display(None)
assert display, 'An X11 display is required'
root = function(X, 'XDefaultRootWindow', Window, Display)(display)
query = function(X, 'XQueryTree', C.c_int, Display, Window, C.POINTER(Window), C.POINTER(Window), C.POINTER(C.POINTER(Window)), C.POINTER(C.c_uint))
fetch_name = function(X, 'XFetchName', C.c_int, Display, Window, C.POINTER(C.c_char_p))
free = function(X, 'XFree', C.c_int, C.c_void_p)
flush = function(X, 'XFlush', C.c_int, Display)
image = function(X, 'XGetImage', C.c_void_p, Display, Window, C.c_int, C.c_int, C.c_uint, C.c_uint, C.c_ulong, C.c_int)
pixel = function(X, 'XGetPixel', C.c_ulong, C.c_void_p, C.c_int, C.c_int)
destroy_image = function(X, 'XDestroyImage', C.c_int, C.c_void_p)
keysym = function(X, 'XStringToKeysym', C.c_ulong, C.c_char_p)
keycode = function(X, 'XKeysymToKeycode', C.c_uint, Display, C.c_ulong)

def find(parent):
    name = C.c_char_p()
    if fetch_name(display, parent, C.byref(name)) and name.value:
        text = name.value.decode(errors='replace')
        free(name)
        matches = text == 'Mimir Viz backend smoke'
    else:
        matches = False
    r, p, children, count = Window(), Window(), C.POINTER(Window)(), C.c_uint()
    if query(display, parent, C.byref(r), C.byref(p), C.byref(children), C.byref(count)):
        nodes = list(children[:count.value])
        if children:
            free(children)
        for child in nodes:
            found = find(child)
            if found:
                return found
    return parent if matches else 0

log_path = Path('/tmp/mimir-native-' + args.backend + '.log')
env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', QT_QPA_PLATFORM='xcb', GDK_BACKEND='x11', GDK_CORE_DEVICE_EVENTS='1', QT_SCALE_FACTOR='1', QT_SCREEN_SCALE_FACTORS='1', QT_FONT_DPI='96', GDK_SCALE='1', GDK_DPI_SCALE='1')
with log_path.open('w') as log:
    process = subprocess.Popen([args.executable, '20'], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 10
        wid = 0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(log_path.read_text())
            wid = find(root)
            if wid and 'PIXELS_OK' in log_path.read_text():
                break
            time.sleep(.1)
        assert wid, log_path.read_text()
        function(X, 'XMoveWindow', C.c_int, Display, Window, C.c_int, C.c_int)(display, wid, 0, 0)
        function(X, 'XMapRaised', C.c_int, Display, Window)(display, wid)
        flush(display)
        time.sleep(.6)
        header = 34 if args.backend in ('QT', 'GTK') else 0
        shot = image(display, wid, 0, header, 320, 240, C.c_ulong(-1).value, 2)
        assert shot, 'Cannot read the native window'
        try:
            for x, y, expected in [(20,30,0xe72a3f),(110,30,0x2dd25a),(250,190,0x5a64f0),(290,220,0x112233)]:
                actual = pixel(shot, x, y) & 0xffffff
                assert actual == expected, (args.backend, x, y, hex(actual), hex(expected))
            assert any((pixel(shot,x,y)&0xffffff)==0xffffff for x in range(10,180) for y in range(120,145)), 'Native text missing'
        finally:
            destroy_image(shot)
        focus=function(X,'XSetInputFocus',C.c_int,Display,Window,C.c_int,C.c_ulong)
        warp=function(X,'XWarpPointer',C.c_int,Display,Window,Window,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_int,C.c_int)
        button=function(T,'XTestFakeButtonEvent',C.c_int,Display,C.c_uint,C.c_int,C.c_ulong)
        key=function(T,'XTestFakeKeyEvent',C.c_int,Display,C.c_uint,C.c_int,C.c_ulong)
        focus(display,wid,1,0)
        warp(display,0,wid,0,0,0,0,20,30+header)
        flush(display);time.sleep(.1)
        for number in [1,4]:
            button(display,number,True,0);button(display,number,False,0)
        for name in ['h','r']:
            code=keycode(display,keysym(name.encode()))
            key(display,code,True,0);key(display,code,False,0)
        flush(display);time.sleep(.5)
        geometry=function(X,'XGetGeometry',C.c_int,Display,Window,C.POINTER(Window),C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint))
        gr,gx,gy,gw,gh,gb,gd=Window(),C.c_int(),C.c_int(),C.c_uint(),C.c_uint(),C.c_uint(),C.c_uint()
        geometry(display,wid,C.byref(gr),C.byref(gx),C.byref(gy),C.byref(gw),C.byref(gh),C.byref(gb),C.byref(gd))
        assert (gw.value,gh.value)==(400,280+header), ('Resize',gw.value,gh.value)
        if header:
            warp(display,0,wid,0,0,0,0,20,17)
            flush(display);time.sleep(.1)
            button(display,1,True,0);button(display,1,False,0)
            flush(display);time.sleep(.2)
        class Data(C.Union):
            _fields_ = [('l', C.c_long * 5), ('b', C.c_char * 20)]
        class Client(C.Structure):
            _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int), ('display', Display), ('window', Window), ('message_type', C.c_ulong), ('format', C.c_int), ('data', Data)]
        class XEvent(C.Union):
            _fields_ = [('client', Client), ('pad', C.c_long * 24)]
        atom = function(X, 'XInternAtom', C.c_ulong, Display, C.c_char_p, C.c_int)
        event = XEvent()
        event.client.type, event.client.display, event.client.window = 33, display, wid
        event.client.message_type = atom(display, b'WM_PROTOCOLS', False)
        event.client.format = 32
        event.client.data.l[0] = atom(display, b'WM_DELETE_WINDOW', False)
        function(X, 'XSendEvent', C.c_int, Display, Window, C.c_int, C.c_long, C.POINTER(XEvent))(display, wid, False, 0, C.byref(event))
        flush(display)
        assert process.wait(timeout=5) == 0, log_path.read_text()
        text = log_path.read_text()
        for expected in ['PIXELS_OK', 'DOWN 20 30', 'UP 20 30', 'WHEEL 1', 'TEXT 104', 'CLOSE']:
            assert expected in text, (expected, text)
        if header:
            assert text.count('KEY 7 ') >= 2, ('Native toolbar not connected', text)
        print(args.backend, 'native primitives, texture, clip, mouse, wheel, keyboard, resize, close' + (', toolbar OK' if header else ' OK'))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
function(X, 'XCloseDisplay', C.c_int, Display)(display)
