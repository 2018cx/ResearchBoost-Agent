"""Read members of a remote .zip over HTTP Range requests (only the central directory and the wanted members move)."""
import io, sys, time, urllib.request, zipfile

class HttpFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
            self.url = r.geturl()            # follow redirects once (hub -> CDN)
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def _get(self, a, b):
        for t in range(6):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={a}-{b - 1}'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == b - a:
                    return data
            except Exception as e:
                err = e
            time.sleep(2 * (t + 1))
        raise IOError(f'range {a}-{b} of {self.url} failed')
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b''
        if n >= self.block:                   # big reads (central directory, members) go straight through
            data = self._get(self.pos, self.pos + n)
        else:
            k = self.pos // self.block
            if k not in self.cache:
                if len(self.cache) > 64:
                    self.cache.clear()
                self.cache[k] = self._get(k * self.block, min(self.size, (k + 1) * self.block))
            off = self.pos - k * self.block
            data = self.cache[k][off:off + n]
            if len(data) < n:
                saved = self.pos
                self.pos += len(data)
                data += self.read(n - len(data))
                self.pos = saved
        self.pos += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)

def open_zip(url):
    return zipfile.ZipFile(HttpFile(url))

if __name__ == '__main__':
    z = open_zip(sys.argv[1])
    names = z.namelist()
    print(len(names), names[:5], names[-3:])
