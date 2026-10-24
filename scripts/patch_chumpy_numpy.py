"""Make chumpy 0.70's legacy scalar exports work with NumPy 1.26."""

from pathlib import Path
import os
import sysconfig
import tempfile


def main():
    source = Path(sysconfig.get_path('purelib')) / 'chumpy' / '__init__.py'
    old = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'
    new = 'from builtins import bool, int, float, complex, object, str\nfrom numpy import nan, inf\nunicode = str'
    text = source.read_text()
    if new in text:
        print('chumpy compatibility patch is already installed')
        return
    if text.count(old) != 1:
        raise RuntimeError('Unexpected chumpy source; refusing to guess a patch')
    # Atomic replacement also avoids mutating other environments via hardlinks.
    fd, filename = tempfile.mkstemp(dir=source.parent, prefix='.hocap-chumpy-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text.replace(old, new))
        os.chmod(filename, source.stat().st_mode)
        os.replace(filename, source)
    finally:
        if os.path.exists(filename):
            os.unlink(filename)
    print(f'Patched {source}')


if __name__ == '__main__':
    main()
