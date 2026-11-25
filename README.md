# VIctory: GTA VI on PC (RX 6800 XT build)

A Windows x64 compatibility layer that runs the Xbox Series build of Grand Theft Auto VI on a PC: the console's own
code runs natively, the layer provides the console's services and translates its GPU work (GFX10 shaders to DXIL)
for Direct3D 12. **You need your own legally obtained copy of the game**, already readable on disk. Nothing here
decrypts anything or works around any protection, and no game files are included or shared.

## Status (26 Nov 2026, tested on my AMD Radeon RX 6800 XT, i7-12700K, 32 GB, Windows 11)

| | Result |
|---|---|
| Boot, intro video, start screen, menus | work |
| Loading into the world | works, no failed DirectStorage reads |
| The world (prologue) | drawn and lit; characters and hair complete; **picture almost complete once loaded**; cutscenes load much more |
| Speed in the world | **4.5-5 fps** with the full scene (about 5,000-6,000 draws a frame), peaks of 6-7, Series X mode, no upscaling |
| Stability | 17 min in the world without a crash or device removal |
| Memory | about 28 GB committed |

Screenshots from my RX 6800 XT (prologue, Series X mode, 2-3 fps shown in the window title):

![The cell at sunset, by the window](docs/screenshots/cell-window.png)
![The cell, sunlight on the wall](docs/screenshots/cell-sink.png)

Still wrong: the title's ray-traced light does not run by default yet (`VICTORY_RAYS=1` is experimental), so a
constant ambient light stands in for it; textures can still pop in late.

## What this build adds

- **Title patches for 1.0.2.1** (in memory, the file is never touched): the title's hang detector, which ended the
  game after ~9 minutes of slow frames, and an audio lock after ~2 minutes. `VICTORY_PATCH_SKIP=<n>,...` drops them.
- **Textures:** uploaded placed textures are refreshed when DirectStorage writes their pages after the upload (walls
  and details that stayed missing for a whole run); higher mips written later reach the card; sharper views are not
  held back when the card is full; XOR-swizzled textures are uploaded; sRGB/SNORM view formats kept on castable
  copies (`VICTORY_TEXTURE_CAST`, on; an A8_UNORM view that removed the device is fixed); planar
  depth-stencil planes and block-compressed sizes that are not multiples of 4 are uploaded.
- **CPU/GPU waits:** settles for title memory changed under shader writes ask the page guards instead of assuming a
  write (`VICTORY_STALE_ASK`, on), and wait only for the submissions concerned (`VICTORY_STALE_PRECISE`, on).
- **Video memory and geometry (on):** pooled upload/readback buffers, kept texture hashes (a texture is checked for
  title writes when the list is recorded, so a rewritten one is uploaded in the same frame), staging buffers dropped
  after rare uploads, LRU eviction only while the card is over its target and never for a texture a view slot still
  names; wave64 mesh shaders exchange first-lane reads across their two 32-lane halves (web triangles and deformed
  faces); read-only raw views off 16-byte alignment get their own copy (writable ones stay on the shared copy); heap
  buffer reads wait for the UAV writes before them (skinned characters). `VICTORY_<NAME>=0` turns each off.
- **Copies of title memory to the card (on):** refreshes of host copies run on several threads
  (`VICTORY_WIDE_REFRESH`), the write-watch journal is read under a shared lock and guarded views are answered
  outside it; only the 4 KiB blocks that changed are sent (`VICTORY_COPY_DIFF`); what the card wrote and was
  copied back is not sent up again unchanged (`VICTORY_COPYBACK_KEEP_COPIED`). Copies went from about 85 to 48 ms
  a frame and from 180 to 70 MB.
- **Textures (on):** title-filled textures get memory only for the mip levels that are mapped
  (`VICTORY_RESIDENT_LEVELS`), and may use 95% of the card's budget (`VICTORY_TEXTURE_ROOM_PERCENT`): the card no
  longer fills up and textures arrive sooner.
- **Shaders:** the translator's optimizer is on again (it no longer drops geometry): GPU time about 110 to 90 ms a
  frame with the same scene.
- **Off by default** (no gain measured, one of them dims reflected light): the wide heap pass (`VICTORY_HEAP_PASS_WIDE`), draw state and table caches (`VICTORY_DRAW_STATE_CACHE`,
  `VICTORY_TABLE_CACHE`); timings with `VICTORY_RECORD_TIMING=1`.
- **Recording the title's commands in parallel (on):** the descriptor heap's made/kept slots are sharded and read
  without locks, the heap buffer view pass is incremental and sharded, made descriptor tables are found again by
  their records (`VICTORY_TABLE_REUSE=fresh`; `=1` also covers the title's own slots and is still wrong for hair),
  recording takes shared locks (`VICTORY_RECORD_PARALLEL`). Lock waits went from about 950 to 36 ms a frame and the
  heap buffer pass from 720-7,300 to about 100 ms.
- **CPU and GPU waits (on):** stale pages are byte-checked before settling the work in flight (29 of 30 settles
  were false alarms), presents go behind the work in flight (`VICTORY_ASYNC_PRESENT`), settles wait only for the
  submissions concerned, and the relay fetches pieces from the card while it runs and merges them on several
  threads (`VICTORY_CARD_FETCH_INFLIGHT`, `VICTORY_RELAY_WIDE_MERGE`).
- **Title memory in place (on, `VICTORY_INPLACE=1`):** view memory the title maps 64 KiB at a time is used by the
  card through a second view of its section when a range is one run (and runs of 4 MiB or more inside a cell with
  holes, `VICTORY_RUN_LEAST_MB`): no copy up, no copy back.
- **Shader translation fixes:** pixel quad reads reach helper lanes (lines along triangles on skin, glass and car
  paint on AMD), helper-lane first-lane reads, normalized buffer conversions, rectangles completed opposite their
  right angle, descriptors joined through nested control flow, ray-traced shadow and water programs translate.
- **Experimental, off by default:** ray tracing (`VICTORY_RAYS=1`: structure builders, ray shaders, ray memory;
  the ambient light steps aside), textures detiled on the card (`VICTORY_GPU_DETILE=1`: right, but too slow while
  it re-detiles whole textures), copy-back on first CPU access (`VICTORY_DEFER_COPYBACK=1`: can hang).
- **Ambient light where ray-traced light does not run:** the output of the title's indirect diffuse resolve
  (`CS_ScreenProbes_TemporalResolve`), which without rays holds garbage (magenta on leaves, cloth and paper, blocks of
  colour indoors), is replaced by a constant (`VICTORY_AMBIENT=<level>`, 0.6 by default; `off` keeps the title's
  output), with a tint (`VICTORY_AMBIENT_TINT=r,g,b`) and less of it in creases and under objects from the scene's
  depth (`VICTORY_AMBIENT_SHADE=strength,radius`). Reflections are held to part of it after the last reflection
  pass (`VICTORY_AMBIENT_REFLECT`, 0.5; without rays they showed the open sky indoors).
- **Indirect draw windows (on):** consecutive indirect records are clamped once per window instead of one by one
  (`VICTORY_INDIRECT_WINDOW=0` turns it off): GPU time about 220 to 150-180 ms a frame here.
- **Release build:** thin LTO and one codegen unit.
- `VICTORY_TITLE_DIR`: the folder with the game's files when the executable is elsewhere.

## Build and run

Requirements: Windows 10/11 x64, an AVX2 CPU, a Direct3D 12 GPU with shader model 6.6 and mesh shaders, Rust.

```powershell
cargo build --release -p victory
```

Pre-build the shaders and pipelines once (about 12 minutes):

```powershell
$env:VICTORY_SKIP_PIPELINES = (Get-Content docs\skip-amd.txt -Raw).Trim()
target\release\victory.exe translate --cache --pipelines --jobs 8 <game dir>\shaders.rpf
```

Play (the settings I test with):

```powershell
$env:VICTORY_SKIP_PIPELINES = (Get-Content docs\skip-amd.txt -Raw).Trim()
$env:VICTORY_LATE_WAIT = '1'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_PIPELINE_WAIT_MS = '24'
$env:VICTORY_RELAY_COPY_QUEUE = '0'
$env:VICTORY_D3D12_DRED = '1'
$env:VICTORY_GAME_DIR = '<game dir>'
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

`docs\skip-amd.txt` names the one shader that hangs the AMD driver's compiler (and leaks memory); it is skipped.
Press A on the start screen. Stay windowed.

## License

GPL-3.0, see [LICENSE](LICENSE). Product names are only used to say what this works with; they belong to their
owners. This project is not affiliated with them.
