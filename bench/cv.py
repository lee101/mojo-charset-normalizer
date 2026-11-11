"""Report measurement noise so SIMD deltas can be judged."""
import sys, time, gc, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import text_scan_case, utf8_case, repeat_to_size
import numpy as np
from mojo_charset_normalizer._lib import lib

def spread(run, reps=25):
    xs=[]
    gc.disable()
    try:
        for _ in range(reps):
            t=time.perf_counter(); run(); xs.append(time.perf_counter()-t)
    finally: gc.enable()
    xs.sort(); med=xs[len(xs)//2]
    return xs[0], med, xs[-1], (xs[-1]-xs[0])/med

ascii_payload = repeat_to_size("A practical English document with ordinary words, punctuation, and 12345.\n","ascii",8_000_000)
utf8_payload = repeat_to_size("Unicode text: Ελληνικά, Русский, العربية, 日本語, français.\n","utf-8",8_000_000)
binp = bytes(range(256))*31_250
for name,(run,) in [("text_scan ascii",(text_scan_case("",ascii_payload)[1],)),
                    ("text_scan utf8",(text_scan_case("",utf8_payload)[1],)),
                    ("text_scan bin",(text_scan_case("",binp)[1],)),
                    ("utf8_stats utf8",(utf8_case("",utf8_payload)[1],))]:
    lo,med,hi,rel = spread(run)
    print(f"{name:18s} min {lo*1000:.3f}  med {med*1000:.3f}  max {hi*1000:.3f} ms  spread {rel*100:.0f}%")
