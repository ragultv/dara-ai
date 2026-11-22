# Flash drafters

`drafter_nano_v3.dflash` is the Dara Flash drafter for **Dara Nano** (Gemma 4 E2B). It is the one the engine looks for (`--flash auto` searches `flash\` next to `dara.exe`, then `build\flash\`).

| | |
|---|---|
| Size | 36,911,148 bytes |
| SHA-256 | `5423c6ec09d0ec063eea32a4d6cf42d44b7027211614b876990d0b72b9b124ef` |
| Trained | 6 epochs, 34,306 sequences (9.0 M tokens); 69.0% held-out next-token accuracy (v2 was about 55%) |
| Measured | Dara Nano with this drafter averages 1.61x llama.cpp decode speed over 16 tasks on a GTX 1650 (plain Dara 1.14x); see `docs/BENCHMARKING.md` |

## Use it

```powershell
# next to the engine, where the installer and the release zip put drafters
copy flash\drafter_nano_v3.dflash bin\flash\
# or from a source checkout
copy flash\drafter_nano_v3.dflash build\flash\
# or name it explicitly
dara.exe serve --model dara-nano --flash flash\drafter_nano_v3.dflash
```

The Core 4B drafter (`drafter_4b_v2.dflash`) is a release asset of `flash-drafters-v2` and is not in this folder. Model files are not stored in git; this one drafter is, so a checkout of `dev`
has the v3 file until the `flash-drafters-v3` GitHub release (which the download catalog and the release workflow expect) is published.
