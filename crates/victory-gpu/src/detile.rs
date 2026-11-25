//! Detiling on the card (`VICTORY_GPU_DETILE=1`; off by default): a title
//! texture's memory goes up as it is, tiled, and a compute program of ours
//! writes its texels as linear rows into a device-local buffer, laid out as
//! the host texture's subresources for a copy; the list then copies that
//! buffer into the texture as it copied the staging buffer before.
//!
//! The program computes each element's address as
//! `victory_formats::texture::Layout::offset` does, from the same pattern
//! tables (`Layout::addressing`), which go up in a header after the tiled
//! bytes. One thread writes one 32-bit word of a linear row: a part of an
//! element of 4 to 16 bytes (block-compressed formats are elements of 8 or
//! 16 bytes), or the 2 or 4 elements of 1 or 2 bytes in it. Every swizzle
//! mode the CPU detiles is covered, with mips, array slices and depth
//! slices; textures with a second (stencil) plane stay on the CPU.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use victory_formats::texture::{ADDRESS_BITS, Dimension, Layout};

/// Whether title textures are detiled on the card (`VICTORY_GPU_DETILE=1`;
/// off by default: the CPU detiles them).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCE.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_DETILE").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// On for this thread's tests alone (`crate::texture`'s).
    pub(crate) static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=18, b0), SRV(t0), UAV(u0)"
cbuffer Sub : register(b0) {
    // Bytes from the tiled memory's start to the mip in slice 0, and
    // between slices; the first array or depth slice; depth slices a slice
    // holds (thick blocks), else 1.
    uint mip_offset;
    uint slice_size;
    uint first;
    uint slab;
    // The mip in elements, and the bytes of one.
    uint width;
    uint height;
    uint n;
    // Elements (linear) or blocks (tiled) per row.
    uint pitch;
    uint tail_x;
    uint tail_y;
    // The subresource in the linear buffer.
    uint dst_offset;
    uint row_pitch;
    uint depth_pitch;
    // Bit 0: linear memory; bit 1: zeros (a level that is not the title's).
    uint flags;
    // Block width, height (elements) and bytes, log2: 8 bits each.
    uint geometry;
    uint xor_bits;
    // Where the pattern tables are, in the tiled buffer.
    uint header;
    // Words a row writes.
    uint row_dwords;
};
ByteAddressBuffer tiled : register(t0);
RWByteAddressBuffer out_rows : register(u0);
static const uint BITS = 14;
groupshared uint flips[3 * BITS];

uint flip(uint axis, uint c) {
    uint acc = 0;
    [unroll]
    for (uint b = 0; b < BITS; b++) {
        if ((c >> b) & 1u) {
            acc ^= flips[axis * BITS + b];
        }
    }
    return acc;
}

uint address(uint x, uint y, uint z) {
    uint base = mip_offset + (z / slab) * slice_size;
    if (flags & 1u) {
        return base + (y * pitch + x) * n;
    }
    uint bw = geometry & 0xffu;
    uint bh = (geometry >> 8) & 0xffu;
    uint bl = (geometry >> 16) & 0xffu;
    x += tail_x;
    y += tail_y;
    uint row = ((y >> bh) * pitch) << bl;
    uint col = ((x >> bw) << bl) | flip(0, x);
    return base + row + (col ^ flip(1, y) ^ flip(2, z) ^ xor_bits);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupThreadID, uint3 id : SV_DispatchThreadID) {
    if (g.x < 3 * BITS) {
        flips[g.x] = (flags & 1u) ? 0u : tiled.Load(header + g.x * 4);
    }
    GroupMemoryBarrierWithGroupSync();
    if (id.x >= row_dwords) {
        return;
    }
    uint at = dst_offset + id.z * depth_pitch + id.y * row_pitch + id.x * 4;
    if (flags & 2u) {
        out_rows.Store(at, 0u);
        return;
    }
    uint byte = id.x * 4;
    uint z = first + id.z;
    uint v = 0;
    if (n >= 4) {
        uint x = byte / n;
        v = tiled.Load(address(x, id.y, z) + (byte - x * n));
    } else {
        uint per = 4 / n;
        uint mask = n == 1 ? 0xffu : 0xffffu;
        for (uint k = 0; k < per; k++) {
            uint x = byte / n + k;
            if (x < width) {
                uint a = address(x, id.y, z);
                v |= ((tiled.Load(a & ~3u) >> ((a & 3u) * 8)) & mask) << (k * n * 8);
            }
        }
    }
    out_rows.Store(at, v);
}
"#;

/// Root constants of one dispatch.
const CONSTANTS: usize = 18;

/// The pattern tables after the tiled bytes: words.
pub(crate) const HEADER_WORDS: usize = 3 * ADDRESS_BITS;

pub(crate) struct Detiler {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Detiler {}
// SAFETY: as above.
unsafe impl Sync for Detiler {}

/// The program, made once; `None` if it does not compile (textures are
/// then detiled on the CPU).
pub(crate) fn detiler() -> Option<&'static Detiler> {
    static DETILER: std::sync::OnceLock<Option<Detiler>> = std::sync::OnceLock::new();
    DETILER
        .get_or_init(|| {
            let device = &crate::host::host()?.device;
            match make_for(device) {
                Ok(d) => Some(d),
                Err(e) => {
                    tracing::warn!(error = %e, "no detiling program: textures are detiled on the CPU");
                    None
                }
            }
        })
        .as_ref()
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Detiler> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("detile"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            let text = String::from_utf8_lossy(text);
            tracing::warn!("detiling program: {text}");
            return Err(windows::core::Error::new(e.code(), text.as_ref()));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Detiler { root, pso })
    }
}

/// Device-local buffers the program writes into, given back when their
/// texture's staging state goes (after every list that used it was
/// reset): made anew for each texture, they cost the submitting thread
/// what the detiling saved while the title streams textures in.
static POOL: std::sync::Mutex<Vec<(ID3D12Resource, u64)>> = std::sync::Mutex::new(Vec::new());
static POOLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The most bytes [`POOL`] holds.
const POOL_BYTES: u64 = 256 << 20;

/// A device-local buffer of at least `size` bytes in the common state, and
/// its capacity: a pooled one up to half again as large, else a new one.
pub(crate) fn take_linear(size: u64) -> Option<(ID3D12Resource, u64)> {
    let size = size.max(1).next_multiple_of(64 << 10);
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, (_, cap))| *cap >= size && *cap <= size + size / 2)
            .min_by_key(|(_, (_, cap))| *cap)
            .map(|(i, _)| i);
        if let Some(i) = best {
            let (buffer, cap) = pool.swap_remove(i);
            POOLED.fetch_sub(cap, std::sync::atomic::Ordering::Relaxed);
            return Some((buffer, cap));
        }
    }
    let buffer = crate::host::host()?.uav_buffer(size)?;
    Some((buffer, size))
}

/// Gives back a buffer of [`take_linear`] no list uses any more.
pub(crate) fn give_linear(buffer: ID3D12Resource, cap: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if POOLED.load(Relaxed) + cap > POOL_BYTES {
        return;
    }
    POOLED.fetch_add(cap, Relaxed);
    POOL.lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((buffer, cap));
}

/// One subresource's dispatch.
#[derive(Debug, Clone)]
struct Sub {
    mip: u32,
    constants: [u32; CONSTANTS],
    /// Words of a row the mip's elements fill, and of a whole row.
    row_dwords: u32,
    pitch_dwords: u32,
    height: u32,
    rows: u32,
    depth: u32,
}

/// How a texture's subresources are detiled: the pattern tables that go
/// after its tiled bytes, and a dispatch for each subresource.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    pub(crate) header: [u32; HEADER_WORDS],
    /// Where the header goes in the tiled buffer (after the tiled bytes,
    /// 4-byte aligned).
    pub(crate) header_offset: u32,
    subs: Vec<Sub>,
}

impl Plan {
    /// The bytes the tiled buffer needs: the tiled memory and the header.
    pub(crate) fn tiled_bytes(&self) -> u64 {
        u64::from(self.header_offset) + (HEADER_WORDS * 4) as u64
    }
}

/// The plan for `layout` detiled into the subresources `footprints` (with
/// `rows` rows each) of a host texture of `mips` levels, as
/// `crate::texture` detiles it on the CPU: subresource `i` is mip
/// `i % mips` of array slice `i / mips` (3D: every depth slice of the mip).
/// `None` where the CPU must do it: no addressing, a footprint smaller than
/// the mip, or offsets past 32 bits.
pub(crate) fn plan(
    layout: &Layout,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    mips: u32,
) -> Option<Plan> {
    let a = layout.addressing()?;
    if layout.desc.samples != 1 || mips == 0 || footprints.len() != rows.len() {
        return None;
    }
    let header_offset = u32::try_from(layout.size.next_multiple_of(4)).ok()?;
    header_offset.checked_add((HEADER_WORDS * 4) as u32)?;
    let mut header = [0u32; HEADER_WORDS];
    for (axis, flips) in a.flips.iter().enumerate() {
        header[axis * ADDRESS_BITS..][..ADDRESS_BITS].copy_from_slice(flips);
    }
    let n = layout.desc.element_bytes;
    let three_d = layout.desc.dimension == Dimension::Tex3d;
    let mut subs = Vec::with_capacity(footprints.len());
    for (i, (fp, &rows)) in footprints.iter().zip(rows).enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let m = layout.mips.get(mip as usize)?;
        let first = if three_d {
            if slice != 0 {
                return None;
            }
            0
        } else {
            slice
        };
        if first >= layout.slices && !three_d {
            return None;
        }
        let f = fp.Footprint;
        let row_pitch = f.RowPitch;
        let depth_pitch = row_pitch.checked_mul(rows)?;
        let bytes = m.width.checked_mul(n)?;
        let need = u64::from(m.depth - 1) * u64::from(depth_pitch)
            + u64::from(m.height - 1) * u64::from(row_pitch)
            + u64::from(bytes);
        let have = u64::from(depth_pitch) * u64::from(f.Depth);
        if row_pitch < bytes
            || need > have
            || !row_pitch.is_multiple_of(4)
            || !fp.Offset.is_multiple_of(4)
        {
            return None;
        }
        u32::try_from(fp.Offset + have).ok()?;
        let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
        let pitch = if a.linear {
            m.pitch
        } else {
            m.pitch >> a.bw_log2
        };
        let mut constants = [0u32; CONSTANTS];
        constants[..17].copy_from_slice(&[
            u32::try_from(m.offset).ok()?,
            u32::try_from(layout.slice_size).ok()?,
            first,
            a.slab,
            m.width,
            m.height,
            n,
            pitch,
            tail_x,
            tail_y,
            fp.Offset as u32,
            row_pitch,
            depth_pitch,
            u32::from(a.linear),
            a.bw_log2 | (a.bh_log2 << 8) | (a.block_log2 << 16),
            a.xor,
            header_offset,
        ]);
        subs.push(Sub {
            mip,
            constants,
            row_dwords: bytes.div_ceil(4),
            pitch_dwords: row_pitch / 4,
            height: m.height,
            rows,
            depth: m.depth.min(f.Depth),
        });
    }
    Some(Plan {
        header,
        header_offset,
        subs,
    })
}

/// Records the detiling of `plan` from the tiled buffer at `tiled` (host
/// GPU address: the tiled bytes, then [`Plan::header`]) into `linear`,
/// which goes from the common state to the copy source state. Levels
/// whose bit is clear in `held` are zeros (levels on demand,
/// `crate::texture`). Replaces the list's compute root signature and
/// pipeline.
pub(crate) fn record(
    d: &Detiler,
    l: &ID3D12GraphicsCommandList,
    tiled: u64,
    linear: &ID3D12Resource,
    plan: &Plan,
    held: u32,
) {
    use crate::host::transition;
    // SAFETY: an open list; our root signature and pipeline; live buffers,
    // the tiled one in an upload heap (always readable), `linear` in the
    // common state and written only here.
    unsafe {
        l.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        l.SetComputeRootSignature(&d.root);
        l.SetPipelineState(&d.pso);
        l.SetComputeRootShaderResourceView(1, tiled);
        l.SetComputeRootUnorderedAccessView(2, linear.GetGPUVirtualAddress());
        for s in &plan.subs {
            let mut c = s.constants;
            let zero = s.mip < 32 && held & (1 << s.mip) == 0;
            let (dwords, rows) = if zero {
                c[13] |= 2;
                (s.pitch_dwords, s.rows)
            } else {
                (s.row_dwords, s.height)
            };
            c[17] = dwords;
            if dwords == 0 || rows == 0 || s.depth == 0 {
                continue;
            }
            l.SetComputeRoot32BitConstants(0, CONSTANTS as u32, c.as_ptr().cast(), 0);
            l.Dispatch(dwords.div_ceil(64), rows, s.depth);
        }
        l.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Pipes, Swizzle};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0};
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    struct Card {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        detiler: Detiler,
    }

    /// The machine's card, or WARP without one (or with
    /// `VICTORY_DETILE_TEST_WARP=1`).
    fn device() -> Option<(ID3D12Device, bool)> {
        let warp = std::env::var_os("VICTORY_DETILE_TEST_WARP").is_some_and(|v| v == "1");
        // SAFETY: plain device creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            if !warp
                && D3D12CreateDevice(
                    None::<&windows::core::IUnknown>,
                    D3D_FEATURE_LEVEL_11_0,
                    &mut device,
                )
                .is_ok()
                && let Some(device) = device
            {
                return Some((device, false));
            }
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            Some((device?, true))
        }
    }

    impl Card {
        fn new() -> Self {
            let (device, warp) = device().expect("a D3D12 device (card or WARP)");
            eprintln!("detiling on {}", if warp { "WARP" } else { "the card" });
            // SAFETY: plain D3D12 setup.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                let event = CreateEventW(None, false, false, None).unwrap();
                let detiler = make_for(&device).expect("the detiling program compiles");
                Card {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    detiler,
                }
            }
        }

        fn buffer(
            &self,
            heap: D3D12_HEAP_TYPE,
            size: u64,
            state: D3D12_RESOURCE_STATES,
        ) -> ID3D12Resource {
            let flags = if heap == D3D12_HEAP_TYPE_DEFAULT {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: heap,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        /// The tiled bytes and the plan's header, in an upload buffer.
        fn tiled(&self, tiled: &[u8], plan: &Plan) -> ID3D12Resource {
            let b = self.buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                plan.tiled_bytes(),
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            // SAFETY: a mapped upload buffer of `tiled_bytes`.
            unsafe {
                let mut data = std::ptr::null_mut();
                b.Map(0, None, Some(&mut data)).unwrap();
                let data = data.cast::<u8>();
                std::ptr::copy_nonoverlapping(tiled.as_ptr(), data, tiled.len());
                std::ptr::copy_nonoverlapping(
                    plan.header.as_ptr().cast::<u8>(),
                    data.add(plan.header_offset as usize),
                    HEADER_WORDS * 4,
                );
                b.Unmap(0, None);
            }
            b
        }

        fn run(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting for and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                self.queue
                    .ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                if let Err(e) = self.device.GetDeviceRemovedReason() {
                    panic!("device removed: {e}");
                }
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        fn read(&self, b: &ID3D12Resource, len: usize) -> Vec<u8> {
            let mut out = vec![0u8; len];
            // SAFETY: a read-back buffer of at least `len` bytes, not in use.
            unsafe {
                let mut data = std::ptr::null_mut();
                b.Map(0, None, Some(&mut data)).unwrap();
                std::ptr::copy_nonoverlapping(data.cast::<u8>(), out.as_mut_ptr(), len);
                b.Unmap(0, None);
            }
            out
        }

        /// Detiles `tiled` on the card into the subresources `footprints`
        /// (`total` bytes) and reads them back.
        fn detile(&mut self, tiled: &[u8], plan: &Plan, total: u64, held: u32) -> Vec<u8> {
            let src = self.tiled(tiled, plan);
            let linear = self.buffer(D3D12_HEAP_TYPE_DEFAULT, total, D3D12_RESOURCE_STATE_COMMON);
            let back = self.buffer(
                D3D12_HEAP_TYPE_READBACK,
                total,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            // SAFETY: an open list and live buffers.
            unsafe {
                record(
                    &self.detiler,
                    &self.list,
                    src.GetGPUVirtualAddress(),
                    &linear,
                    plan,
                    held,
                );
                self.list
                    .CopyBufferRegion(&back, 0, &linear, 0, total.max(256));
            }
            self.run();
            self.read(&back, total as usize)
        }
    }

    /// Linear footprints as a host texture's subresources would have them:
    /// rows 256-byte aligned, subresources 512-byte aligned.
    fn footprints(l: &Layout) -> (Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, Vec<u32>, u64) {
        let d = l.desc;
        let n = d.element_bytes;
        let slices = if d.dimension == Dimension::Tex3d {
            1
        } else {
            d.depth
        };
        let (mut fps, mut rows, mut at) = (vec![], vec![], 0u64);
        for _ in 0..slices {
            for m in &l.mips {
                let pitch = (m.width * n).next_multiple_of(256);
                at = at.next_multiple_of(512);
                fps.push(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: at,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_UNKNOWN,
                        Width: m.width,
                        Height: m.height,
                        Depth: m.depth,
                        RowPitch: pitch,
                    },
                });
                rows.push(m.height);
                at += u64::from(pitch) * u64::from(m.height) * u64::from(m.depth);
            }
        }
        (fps, rows, at)
    }

    /// Bytes from a seed, for tiled memory.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// Compares the card's rows with `Layout::detile`'s for every
    /// subresource; levels not in `held` must be zeros through their
    /// whole footprint.
    fn check(card: &mut Card, d: Desc, pipes: Pipes, pbx: u32, held: u32) -> usize {
        let l = Layout::for_gpu(&d, pipes, pbx).unwrap();
        let (fps, rows, total) = footprints(&l);
        let mips = d.mips;
        let p = plan(&l, &fps, &rows, mips).unwrap_or_else(|| panic!("no plan for {d:?}"));
        let tiled = noise(
            l.size as usize,
            u64::from(d.swizzle.0) * 977 + u64::from(d.element_bytes) + u64::from(d.width),
        );
        let got = card.detile(&tiled, &p, total, held);
        let n = d.element_bytes as usize;
        for (i, fp) in fps.iter().enumerate() {
            let (mip, slice) = (i as u32 % mips, i as u32 / mips);
            let m = l.mips[mip as usize];
            let f = fp.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows[i] as usize * f.Depth as usize;
            let at = fp.Offset as usize;
            let gpu = &got[at..at + len];
            if mip < 32 && held & (1 << mip) == 0 {
                assert!(
                    gpu.iter().all(|&b| b == 0),
                    "{d:?} mip {mip} slice {slice}: not zeros"
                );
                continue;
            }
            let mut want = vec![0u8; len];
            l.detile(
                &tiled,
                mip,
                slice,
                &mut want,
                pitch,
                pitch * rows[i] as usize,
            )
            .unwrap();
            for z in 0..m.depth as usize {
                for y in 0..m.height as usize {
                    let r = z * pitch * rows[i] as usize + y * pitch;
                    let w = m.width as usize * n;
                    assert_eq!(
                        gpu[r..r + w],
                        want[r..r + w],
                        "{d:?} pipes {pipes:?} xor {pbx:#x}: mip {mip} slice {slice} row {y} depth {z}"
                    );
                }
            }
        }
        fps.len()
    }

    #[allow(clippy::too_many_arguments)]
    fn desc(
        w: u32,
        h: u32,
        depth: u32,
        mips: u32,
        bpe: u32,
        block: u32,
        swizzle: Swizzle,
        three_d: bool,
    ) -> Desc {
        Desc {
            dimension: if three_d {
                Dimension::Tex3d
            } else {
                Dimension::Tex2d
            },
            width: w,
            height: h,
            depth,
            mips,
            element_bytes: bpe,
            block,
            samples: 1,
            swizzle,
        }
    }

    /// Every swizzle mode the CPU detiles, every element size (8 to 128
    /// bits, block compression of 8 and 16 bytes), mip chains with and
    /// without a tail, array slices and 3D textures: the card's rows are
    /// the CPU's, byte for byte.
    #[test]
    fn the_card_detiles_as_the_cpu_does() {
        let mut card = Card::new();
        let modes = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27];
        let formats = [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (8, 4), (16, 4)];
        let mut subresources = 0;
        for mode in modes {
            let swizzle = Swizzle(mode);
            let gpus: &[(Pipes, u32)] = if swizzle.xor() {
                &[
                    (Pipes::SERIES_X, 0),
                    (Pipes::SERIES_X, 0x1f3),
                    (Pipes::SERIES_S, 0x2d),
                ]
            } else {
                &[(Pipes::SERIES_X, 0)]
            };
            for &(bpe, block) in &formats {
                let texels = block;
                let mut cases = vec![
                    // A full chain with a tail, odd sizes.
                    desc(75 * texels, 41 * texels, 1, 7, bpe, block, swizzle, false),
                    // Array slices, a few mips.
                    desc(33 * texels, 66 * texels, 3, 4, bpe, block, swizzle, false),
                    // One level larger than a 64 KiB block.
                    desc(300, 260, 1, 1, bpe, block, swizzle, false),
                ];
                if matches!(mode, 0 | 5 | 9 | 17 | 21 | 24..=27) {
                    cases.push(desc(
                        20 * texels,
                        18 * texels,
                        11,
                        5,
                        bpe,
                        block,
                        swizzle,
                        true,
                    ));
                }
                for d in cases {
                    for &(pipes, pbx) in gpus {
                        subresources += check(&mut card, d, pipes, pbx, u32::MAX);
                    }
                }
            }
        }
        eprintln!("{subresources} subresources compared");
    }

    /// Levels that are not the title's are zeros through their footprint;
    /// the others are detiled.
    #[test]
    fn levels_not_held_are_zeros() {
        let mut card = Card::new();
        for (swizzle, bpe, block) in [
            (Swizzle::SW_64KB_R_X, 4, 1),
            (Swizzle::SW_4KB_S, 16, 4),
            (Swizzle::LINEAR, 1, 1),
        ] {
            let d = desc(130, 90, 2, 8, bpe, block, swizzle, false);
            check(&mut card, d, Pipes::SERIES_X, 0x7, 0b1010_0110);
        }
    }

    /// Through a real host texture: the device's own footprints (BC and
    /// uncompressed, odd sizes, an array), the linear buffer copied into
    /// the texture and back, equal to the CPU's rows.
    #[test]
    fn the_card_detiles_into_host_textures() {
        let mut card = Card::new();
        let cases = [
            (
                DXGI_FORMAT_BC1_UNORM,
                8,
                4,
                Swizzle::SW_64KB_S_X,
                1000,
                600,
                1,
                10,
            ),
            (
                DXGI_FORMAT_BC7_UNORM,
                16,
                4,
                Swizzle::SW_64KB_R_X,
                256,
                128,
                2,
                9,
            ),
            (
                DXGI_FORMAT_BC3_UNORM,
                16,
                4,
                Swizzle::SW_4KB_S,
                70,
                30,
                1,
                5,
            ),
            (
                DXGI_FORMAT_R8G8B8A8_UNORM,
                4,
                1,
                Swizzle::SW_64KB_D_X,
                123,
                77,
                3,
                7,
            ),
            (
                DXGI_FORMAT_R8_UNORM,
                1,
                1,
                Swizzle::SW_64KB_Z_X,
                99,
                45,
                1,
                7,
            ),
            (
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                8,
                1,
                Swizzle::SW_64KB_S,
                64,
                64,
                6,
                7,
            ),
            (
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                16,
                1,
                Swizzle::LINEAR,
                40,
                20,
                1,
                6,
            ),
        ];
        for (format, bpe, block, swizzle, w, h, array, mips) in cases {
            let d = desc(w, h, array, mips, bpe, block, swizzle, false);
            let l = Layout::for_gpu(&d, Pipes::SERIES_X, 0x55).unwrap();
            let td = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: array as u16,
                MipLevels: mips as u16,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let count = mips * array;
            let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
            let mut rows = vec![0u32; count as usize];
            let mut total = 0;
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: outputs for `count` subresources; plain creation.
            unsafe {
                card.device.GetCopyableFootprints(
                    &td,
                    0,
                    count,
                    0,
                    Some(fps.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    None,
                    Some(&mut total),
                );
                card.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &td,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut texture,
                    )
                    .unwrap();
            }
            let texture = texture.unwrap();
            let p = plan(&l, &fps, &rows, mips).unwrap_or_else(|| panic!("no plan for {d:?}"));
            let tiled = noise(l.size as usize, u64::from(w) * 31 + u64::from(h));
            let src = card.tiled(&tiled, &p);
            let linear = card.buffer(D3D12_HEAP_TYPE_DEFAULT, total, D3D12_RESOURCE_STATE_COMMON);
            let back = card.buffer(
                D3D12_HEAP_TYPE_READBACK,
                total,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let location =
                |r: &ID3D12Resource, i: usize, placed: bool| D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: a borrowed pointer the location never releases.
                    pResource: unsafe { std::mem::transmute_copy(r) },
                    Type: if placed {
                        D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
                    } else {
                        D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
                    },
                    Anonymous: if placed {
                        D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: fps[i],
                        }
                    } else {
                        D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: i as u32,
                        }
                    },
                };
            // SAFETY: an open list and live resources in the states named.
            unsafe {
                record(
                    &card.detiler,
                    &card.list,
                    src.GetGPUVirtualAddress(),
                    &linear,
                    &p,
                    u32::MAX,
                );
                for i in 0..fps.len() {
                    card.list.CopyTextureRegion(
                        &location(&texture, i, false),
                        0,
                        0,
                        0,
                        &location(&linear, i, true),
                        None,
                    );
                }
                card.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                for i in 0..fps.len() {
                    card.list.CopyTextureRegion(
                        &location(&back, i, true),
                        0,
                        0,
                        0,
                        &location(&texture, i, false),
                        None,
                    );
                }
            }
            card.run();
            let got = card.read(&back, total as usize);
            let n = bpe as usize;
            for (i, fp) in fps.iter().enumerate() {
                let (mip, slice) = (i as u32 % mips, i as u32 / mips);
                let m = l.mips[mip as usize];
                let pitch = fp.Footprint.RowPitch as usize;
                let len = pitch * rows[i] as usize;
                let mut want = vec![0u8; len];
                l.detile(&tiled, mip, slice, &mut want, pitch, len).unwrap();
                let at = fp.Offset as usize;
                for y in 0..m.height as usize {
                    let r = y * pitch;
                    let w = m.width as usize * n;
                    assert_eq!(
                        got[at + r..at + r + w],
                        want[r..r + w],
                        "{format:?} mip {mip} slice {slice} row {y}"
                    );
                }
            }
        }
    }
}
