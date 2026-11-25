# Ray tracing in this fork: what the title does and what has been built

Status, 2026-11-25: **opt-in behind runtime switches (default off).** The BVH structure builders, address-space directory, descriptor address tables, `DispatchX` thread sizing, and ray intersection/traversal lowering are implemented behind `VICTORY_RAYS=1` (and granular per-subsystem switches) with unit tests on synthetic BVHs and synthetic heaps/lists. End-to-end world lighting still needs in-game validation under those switches.

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun are ray traced. Without them, anything the sun
does not hit directly is black: interiors, the shadowed side of a face, rooms
lit only by lamps.

The title does not use a ray-tracing API:

- Traversal is ordinary compute-shader code: a per-thread stack in shared
  memory, and loads through raw 40-bit addresses of title memory.
- The static world's acceleration structures ship pre-built in the title's
  data, in the console GPU's native node format (64-byte nodes: box nodes with
  four children and 16-bit float bounds, 128-byte box nodes with 32-bit float
  bounds, triangle nodes, instance nodes). They are unpacked on the GPU.
- A set of builder programs (from the title's `XboxBvhBuilder.dll`, 52 compute
  programs) builds and refits the dynamic structures on the GPU every frame.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the title never
describes to it, and the traversal loops live in 64 of the title's own
programs. This fork runs the title's builder and traversal as translated
shaders and emulates the one instruction.

## What this fork adds

1. **The builder programs translate** (49 of 52; all that the title creates in
   the scenes tested). They pass buffer addresses in a spilled parameter table
   rather than in descriptors. Added: the spilled user-data table, root
   unordered-access views (translator and host binding), byte and 16-bit buffer
   accesses, and several control-flow shapes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to the layer's host copy of
   that memory; loads, stores, atomic add/sub and float min/max go through it.
   Before a dispatch the host makes ready the memory its root views and root
   constants name.
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (59 of 64 trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **Host safety, lifetimes, and switches.**
   - `VICTORY_RAYS=1` enables the full ray-tracing stack by default across four switches: `VICTORY_RAY_SHADERS`, `VICTORY_RAY_MEMORY`, `VICTORY_RAY_DISPATCH`, and `VICTORY_BVH_BUILDER` (each inherits `VICTORY_RAYS`, default `0`, and can be toggled individually with `0`/`1`).
   - When ray tracing is enabled and the builder has not tripped off, `ambient` (`CS_ScreenProbes_TemporalResolve` stand-in and `CS_UpdateHistory` reflection clamp) steps aside automatically (`VICTORY_AMBIENT=off` behavior unless `VICTORY_AMBIENT` is explicitly overridden) so the title's resolve output is used.
   - The builder's stages run only while every created builder pipeline has a host pipeline (`VICTORY_BVH_BUILDER=1` / `VICTORY_RAYS=1`; `VICTORY_BVH_QUARANTINE=1` keeps builder stores out of title memory).
   - Precise shader write tracking (`written_root_uavs`, `writes_guest_memory`), writeback ownership intervals across virtual memory remappings/reassignments, retired resource grace lifetimes, owner-aware descriptor/root-slot rebinding, growing dynamic raw-page view runs, and `DispatchX` thread-to-group conversion with tail-lane masking keep builder and traversal passes isolated from unrelated buffers.

## Known gaps

- `VICTORY_FIRSTLANE_OWN=1` and `VICTORY_SKIP_PIPELINES=""` (the default skip list suppresses `39f0e9aeb156a61c`) can affect specific compute passes; in diagnostics builds, `VICTORY_BVH_TRACE=<hash[,hash...]>` logs builder root arguments.
- An access to a title page that has no host copy is lost (a read gives zero, a store is dropped), though ray queries/stores also surface guest faults and dynamic descriptor address tables (`SLOT_VIEW_ADDRESSES`).
- Cost and visual convergence across all hours/interiors on target AMD hardware remain to be measured in game.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
