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

It reads out-of-line definitions (Class::fn and Class<T>::fn, in .cpp and
.h files, so template impl headers count) and inline in-class definitions.

ponytail: a regex scan, not a C++ parser. Other nvcc-only access errors (a
private nested type used as a kernel argument) need the CUDA build itself
(build-cuda/env.sh).
"""
import glob
import os
import re
import sys

LAMBDA = r'KOKKOS_(CLASS_)?LAMBDA'

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
        m = re.search(r'\b(class|struct)\s+' + cls + r'\b(?!\s*[>,=])[^;{]*\{', t)
        if not m:
            continue
        body = t[m.end():block(t, m.end())]
        d = re.search(r'[\s*&~]' + re.escape(fn) + r'\s*\(', body)
        if not d:
            continue
        accs = re.findall(r'\b(public|protected|private)\s*:', body[:d.start()])
        return accs[-1] if accs else ('private' if m.group(1) == 'class' else 'public')
    return None


def inline_defs(t):
    """(class, function, access, offset) of each in-class definition with a
    device lambda. Nested classes are found as classes of their own."""
    item = re.compile(r'\b(public|protected|private)\s*:|(~?\w+)\s*\([^;{}]*\)[\w\s]*\{|\{')
    for m in re.finditer(r'\b(class|struct)\s+(\w+)\b(?!\s*[>,=])[^;{()]*\{', t):
        end = block(t, m.end()) - 1
        acc = 'private' if m.group(1) == 'class' else 'public'
        i = m.end()
        while True:
            s = item.search(t, i, end)
            if not s:
                break
            if s.group(1):
                acc, i = s.group(1), s.end()
                continue
            i = block(t, s.end())
            if s.group(2) and re.search(LAMBDA, t[s.end():i]):
                yield m.group(2), s.group(2), acc, s.start()


def check(root):
    hfiles = glob.glob(os.path.join(root, '*.h')) + glob.glob(os.path.join(root, '..', '*.h'))
    headers = [strip_comments(open(h).read()) for h in hfiles]
    bad = []
    line = lambda t, i: t[:i].count(chr(10)) + 1
    for c in sorted(glob.glob(os.path.join(root, '*.cpp')) + glob.glob(os.path.join(root, '*.h'))):
        t = strip_comments(open(c).read())
        for m in re.finditer(r'\b(\w+)(?:<[^<>;{}]*>)?::(~?\w+)\s*\([^;{]*?\)\s*(const)?\s*\{', t):
            end = block(t, m.end())
            if not re.search(LAMBDA, t[m.end():end]):
                continue
            a = access(headers, m.group(1), m.group(2))
            if a in ('private', 'protected'):
                bad.append(f"{m.group(1)}::{m.group(2)} is {a}: {c}:{line(t, m.start())}")
        if c.endswith('.h'):
            for cls, fn, a, i in inline_defs(t):
                if a in ('private', 'protected'):
                    bad.append(f"{cls}::{fn} is {a}: {c}:{line(t, i)}")
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
        open(os.path.join(d, 'b.h'), 'w').write(
            "template <class B> class T : public B {\n public:\n  void ok() { f(KOKKOS_LAMBDA(int) {}); }\n"
            "  struct In { int x; };\n protected:\n  void inl() { f(KOKKOS_LAMBDA(int) {}); }\n  void tpl();\n};\n"
            "template <class B> void T<B>::tpl() { f(KOKKOS_LAMBDA(int) {}); }\n")
        found = sorted(check(d))
        assert [f.split(' is ')[0] for f in found] == ['A::bad', 'T::inl', 'T::tpl'], found
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
