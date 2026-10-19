#!/usr/bin/env python3

# copy SPPARKS src/libspparks.so and spparks.py to system dirs

instructions = """
Syntax: python install.py [-h] [libdir] [pydir]
        libdir = target dir for src/libspparks.so, default = /usr/local/lib
        pydir = target dir for spparks.py, default = Python site-packages dir
"""

import os, shutil, sys, sysconfig

if (len(sys.argv) > 1 and sys.argv[1] == "-h") or len(sys.argv) > 3:
  print(instructions)
  sys.exit()

if len(sys.argv) >= 2: libdir = sys.argv[1]
else: libdir = "/usr/local/lib"

if len(sys.argv) == 3: pydir = sys.argv[2]
else: pydir = ""

# paths relative to this script, so it works when invoked via make in src dir

here = os.path.dirname(os.path.abspath(__file__))

# copy C lib to libdir if it exists
# warn if not in LD_LIBRARY_PATH or LD_LIBRARY_PATH is undefined

if not os.path.isdir(libdir):
  print("ERROR: libdir %s does not exist" % libdir)
  sys.exit()

if "LD_LIBRARY_PATH" not in os.environ:
  print("WARNING: LD_LIBRARY_PATH undefined, cannot check libdir %s" % libdir)
else:
  libpaths = os.environ['LD_LIBRARY_PATH'].split(':')
  if libdir not in libpaths:
    print("WARNING: libdir %s not in LD_LIBRARY_PATH" % libdir)

lib = os.path.join(here, "..", "src", "libspparks.so")
print("cp %s %s" % (lib, libdir))
shutil.copy(lib, libdir)

# copy spparks.py to pydir, or by default to this Python's site-packages
# (distutils, which the Python 2 version used, is gone from Python 3.12)

if pydir:
  if not os.path.isdir(pydir):
    print("ERROR: pydir %s does not exist" % pydir)
    sys.exit()
else:
  pydir = sysconfig.get_paths()["purelib"]
  print("installing spparks.py in Python site-packages dir")

print("cp %s %s" % (os.path.join(here, "spparks.py"), pydir))
shutil.copy(os.path.join(here, "spparks.py"), pydir)
