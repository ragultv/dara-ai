"""Generate src/tokenizer/unicode_data.inc: the Unicode data the Qwen3.5 tokenizer needs, compiled into the runtime.

  * \\p{L}, \\p{M}, \\p{N} as sorted codepoint ranges (the pre-tokenizer pattern's classes)
  * canonical decompositions (fully expanded, Hangul syllables excluded: they are algorithmic), canonical combining
    classes, and the primary composition pairs (composition exclusions removed) -- what NFC needs
Before writing, the generator runs the SAME NFC algorithm the C++ implements (from these tables) against unicodedata on
every codepoint and on random sequences, and aborts on any difference.
Source of truth: Python's unicodedata (printed version). Usage: python -I tools/gen_unicode_tables.py
"""
import random
import sys
import unicodedata as ud

SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT
MAX = 0x110000


def ranges(pred):
    out, start = [], None
    for cp in range(MAX):
        if 0xD800 <= cp <= 0xDFFF:
            ok = False
        else:
            ok = pred(cp)
        if ok and start is None:
            start = cp
        if not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX - 1))
    return out


def full_decomp(cp):
    d = ud.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return [cp]
    res = []
    for h in d.split():
        res += full_decomp(int(h, 16))
    return res


def main():
    cat = lambda cp: ud.category(chr(cp))
    L = ranges(lambda cp: cat(cp)[0] == "L")
    M = ranges(lambda cp: cat(cp)[0] == "M")
    N = ranges(lambda cp: cat(cp)[0] == "N")

    decomp, ccc = {}, {}
    for cp in range(MAX):
        if 0xD800 <= cp <= 0xDFFF or SBASE <= cp < SBASE + SCOUNT:
            continue
        c = ud.combining(chr(cp))
        if c:
            ccc[cp] = c
        d = full_decomp(cp)
        if d != [cp]:
            decomp[cp] = d
    # primary composites: canonical 2-element decompositions that NFC really recomposes
    comp = []
    for cp in range(MAX):
        if 0xD800 <= cp <= 0xDFFF or SBASE <= cp < SBASE + SCOUNT:
            continue
        d = ud.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = [int(h, 16) for h in d.split()]
        if len(parts) == 2 and ud.normalize("NFC", chr(parts[0]) + chr(parts[1])) == chr(cp):
            comp.append((parts[0], parts[1], cp))
    comp.sort()
    comp_map = {(a, b): c for a, b, c in comp}

    # ---- the algorithm the C++ implements -----------------------------------------------------
    def nfc(s):
        cps = []
        for ch in s:
            cp = ord(ch)
            if SBASE <= cp < SBASE + SCOUNT:
                si = cp - SBASE
                cps += [LBASE + si // NCOUNT, VBASE + (si % NCOUNT) // TCOUNT]
                if si % TCOUNT:
                    cps.append(TBASE + si % TCOUNT)
            else:
                cps += decomp.get(cp, [cp])
        # canonical ordering: stable sort of each run of non-starters by ccc
        i = 0
        while i < len(cps):
            if ccc.get(cps[i], 0) == 0:
                i += 1
                continue
            j = i
            while j < len(cps) and ccc.get(cps[j], 0) != 0:
                j += 1
            cps[i:j] = sorted(cps[i:j], key=lambda c: ccc[c])
            i = j
        # composition
        out = []
        starter_idx = -1
        last_ccc = -1
        for cp in cps:
            c = ccc.get(cp, 0)
            if starter_idx >= 0:
                st = out[starter_idx]
                blocked = last_ccc != -1 and (last_ccc >= c or last_ccc == 0) if len(out) - 1 > starter_idx else False
                if not blocked:
                    comp_cp = None
                    if LBASE <= st < LBASE + LCOUNT and VBASE <= cp < VBASE + VCOUNT:
                        comp_cp = SBASE + ((st - LBASE) * VCOUNT + (cp - VBASE)) * TCOUNT
                    elif SBASE <= st < SBASE + SCOUNT and (st - SBASE) % TCOUNT == 0 and TBASE < cp < TBASE + TCOUNT:
                        comp_cp = st + (cp - TBASE)
                    else:
                        comp_cp = comp_map.get((st, cp))
                    if comp_cp is not None:
                        out[starter_idx] = comp_cp
                        continue
            if c == 0:
                starter_idx = len(out)
                last_ccc = -1
            else:
                last_ccc = c
            out.append(cp)
            if c == 0 and starter_idx == len(out) - 1:
                last_ccc = -1
        return "".join(map(chr, out))

    # ---- self-check against unicodedata -----------------------------------------------------------
    bad = 0
    for cp in range(MAX):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        s = chr(cp)
        if nfc(s) != ud.normalize("NFC", s):
            bad += 1
            if bad < 5:
                print("MISMATCH single", hex(cp))
    rng = random.Random(7)
    pool = [cp for cp in list(ccc) + list(decomp) + [c for _, _, c in comp] + list(range(0x41, 0x7B))
            + list(range(SBASE, SBASE + 400)) + list(range(LBASE, LBASE + 20)) + list(range(VBASE, VBASE + 22)) + list(range(TBASE + 1, TBASE + 28))]
    for _ in range(60000):
        s = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 8)))
        if nfc(s) != ud.normalize("NFC", s):
            bad += 1
            if bad < 8:
                print("MISMATCH seq", [hex(ord(c)) for c in s], [hex(ord(c)) for c in nfc(s)], [hex(ord(c)) for c in ud.normalize("NFC", s)])
    if bad:
        sys.exit(f"{bad} mismatches against unicodedata: aborting")
    print(f"self-check passed: NFC from tables == unicodedata on all {MAX - 2048} codepoints and 60000 random sequences")

    # ---- emit ----------------------------------------------------------------------------------
    pool_cps, index = [], []
    for cp in sorted(decomp):
        index.append((cp, len(pool_cps), len(decomp[cp])))
        pool_cps += decomp[cp]
    ccc_runs, items = [], sorted(ccc.items())
    for cp, c in items:
        if ccc_runs and ccc_runs[-1][1] + 1 == cp and ccc_runs[-1][2] == c:
            ccc_runs[-1][1] = cp
        else:
            ccc_runs.append([cp, cp, c])

    def arr2(name, rs):
        return f"static const uint32_t {name}[][2] = {{\n" + ",\n".join("    {0x%X, 0x%X}" % r for r in rs) + "\n};\n"

    with open("src/tokenizer/unicode_data.inc", "w") as f:
        f.write(f"// GENERATED by tools/gen_unicode_tables.py from Python unicodedata {ud.unidata_version}. Do not edit.\n")
        f.write(arr2("kLetterRanges", L)); f.write(arr2("kMarkRanges", M)); f.write(arr2("kNumberRanges", N))
        f.write("struct DecompEntry { uint32_t cp; uint32_t offset; uint32_t len; };\n")
        f.write("static const DecompEntry kDecompIndex[] = {\n" + ",\n".join("    {0x%X, %d, %d}" % e for e in index) + "\n};\n")
        f.write("static const uint32_t kDecompPool[] = {\n" + ",".join("0x%X" % c for c in pool_cps) + "\n};\n")
        f.write("struct CccRun { uint32_t lo, hi; uint8_t ccc; };\n")
        f.write("static const CccRun kCccRuns[] = {\n" + ",\n".join("    {0x%X, 0x%X, %d}" % tuple(r) for r in ccc_runs) + "\n};\n")
        f.write("struct ComposePair { uint32_t a, b, c; };\n")
        f.write("static const ComposePair kCompose[] = {\n" + ",\n".join("    {0x%X, 0x%X, 0x%X}" % c for c in comp) + "\n};\n")
    print(f"unicodedata {ud.unidata_version}: L {len(L)} ranges, M {len(M)}, N {len(N)}; decompositions {len(index)} (pool {len(pool_cps)}); "
          f"ccc runs {len(ccc_runs)}; compositions {len(comp)}")


if __name__ == "__main__":
    main()
