#!/usr/bin/env python3
"""Find member functions that define a device lambda but are not public.

nvcc rejects an extended __host__ __device__ lambda (KOKKOS_LAMBDA,
KOKKOS_CLASS_LAMBDA) inside a private or protected member function:

  The enclosing parent function ("...") for an extended __host__ __device__
  lambda cannot have private or protected access within its class

HIP and OpenMP accept it, so no local (gfx1151) build catches it; only a CUDA
build does. Run this before committing Kokkos code:

  python3 tools/check_kokkos_lambda_access.py [src/KOKKOS]

Exits 1 and lists each offender (class::function, access, file:line). Fix by
declaring the function in a public section.

ponytail: a regex scan, not a C++ parser: it reads class bodies and
out-of-line definitions (Class::fn(...) { ... }); a lambda in an inline
in-class definition or a nested class is not checked.
"""
import glob
import os
import re
import sys


def strip_comments(t):
    t = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), t, flags=re.S)
    return re.sub(r'//[^\n]*', '', t)


def block(t, i):
    """Index just past the brace block whose opening brace precedes i."""
    depth = 1
    while i < len(t) and depth:
        depth += {'{': 1, '}': -1}.get(t[i], 0)
        i += 1
    return i


def access(headers, cls, fn):
    for t in headers:
        m = re.search(r'\b(class|struct)\s+' + cls + r'\b[^;{]*\{', t)
        if not m:
            continue
        body = t[m.end():block(t, m.end())]
        d = re.search(r'[\s*&~]' + re.escape(fn) + r'\s*\(', body)
        if not d:
            continue
        accs = re.findall(r'\b(public|protected|private)\s*:', body[:d.start()])
        return accs[-1] if accs else ('private' if m.group(1) == 'class' else 'public')
    return None


def check(root):
    headers = [strip_comments(open(h).read())
               for h in glob.glob(os.path.join(root, '*.h')) + glob.glob(os.path.join(root, '..', '*.h'))]
    bad = []
    for c in sorted(glob.glob(os.path.join(root, '*.cpp'))):
        t = strip_comments(open(c).read())
        for m in re.finditer(r'\b(\w+)::(~?\w+)\s*\([^;{]*?\)\s*(const)?\s*\{', t):
            end = block(t, m.end())
            if not re.search(r'KOKKOS_(CLASS_)?LAMBDA', t[m.end():end]):
                continue
            a = access(headers, m.group(1), m.group(2))
            if a in ('private', 'protected'):
                bad.append(f"{m.group(1)}::{m.group(2)} is {a}: {c}:{t[:m.start()].count(chr(10)) + 1}")
    return bad


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'a.h'), 'w').write(
            "class A {\n public:\n  void ok();\n protected:\n  void bad();\n};\n")
        open(os.path.join(d, 'a.cpp'), 'w').write(
            "// A::bad() would be flagged if this comment were code: KOKKOS_LAMBDA\n"
            "void A::ok() { parallel_for(1, KOKKOS_LAMBDA(int i) {}); }\n"
            "void A::bad() { parallel_for(1, KOKKOS_LAMBDA(int i) {}); }\n")
        found = check(d)
        assert len(found) == 1 and found[0].startswith('A::bad is protected'), found
    print("selftest ok")


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        sys.exit(selftest())
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'src', 'KOKKOS')
    bad = check(root)
    for b in bad:
        print(b)
    print(f"{len(bad)} non-public member function(s) define device lambdas (nvcc rejects them)")
    sys.exit(1 if bad else 0)
