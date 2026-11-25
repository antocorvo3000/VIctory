//! Resident-level host copies (`VICTORY_RESIDENT_LEVELS=1`; off by default).
//!
//! Streamed title textures often have their smaller mip levels mapped in title
//! memory while their larger levels are still uncommitted (or belong to another
//! texture). Making a committed host copy of the full mip chain allocates
//! gigabytes of card memory for unmapped levels.
//!
//! When enabled, a host copy of a title-filled texture is created with only the
//! contiguous tail of mip levels whose title memory is mapped: the largest
//! resident level (`first_level`) becomes the host resource's mip `0`, shader
//! views shift `MostDetailedMip`/`MipLevels`/`ResourceMinLODClamp` accordingly,
//! copies translate or skip subresources, and when the title maps larger levels
//! later (`texture_slots_remapped`, `page_written`), a new host copy with the
//! larger levels is allocated, uploaded, swapped into descriptor slots, and the
//! old copy is retired.

use std::sync::atomic::AtomicU64;

use windows::Win32::Graphics::Direct3D12::*;

/// Since start: trimmed host copies created (`first_level > 0`), estimated
/// card bytes saved at creation, host copies grown when larger levels mapped,
/// host copies shrunk when larger levels unmapped, host copies dropped when all
/// levels unmapped.
pub(crate) static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

#[cfg(test)]
thread_local! {
    /// Forces `VICTORY_RESIDENT_LEVELS=1` on the current test thread only.
    pub(crate) static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether resident-level trimming is enabled (`VICTORY_RESIDENT_LEVELS=1`,
/// off by default).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCE.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESIDENT_LEVELS").is_none_or(|v| v != "0"))
}

/// Finds the largest resident mip level (`first_level`, `0` = full chain) such
/// that every level in `first_level..mips` has its title memory mapped (in all
/// slices) and lies before `limit`. Returns `None` if no contiguous suffix from
/// the smallest level up is mapped.
pub(crate) fn resident_first_level(
    layout: &victory_formats::texture::Layout,
    memory: *const u8,
    limit: u64,
) -> Option<u32> {
    let mips = layout.mips.len();
    if mips == 0 || mips > 32 || layout.size == 0 {
        return Some(0);
    }
    if limit == u64::MAX && crate::texture::committed(memory, layout.size as usize) {
        return Some(0);
    }
    let tail = (layout.first_tail_mip as usize).min(mips);
    let there = |from: u64, to: u64| {
        if to <= from {
            return false;
        }
        let len = (to - from) as usize;
        (0..u64::from(layout.slices)).all(|slice| {
            let base = slice * layout.slice_size;
            base.saturating_add(to) <= limit
                && crate::texture::committed(memory.wrapping_add((base + from) as usize), len)
        })
    };
    if tail < mips {
        let tail_end = layout.alignment.min(layout.slice_size);
        if !there(0, tail_end) {
            return None;
        }
        let mut first = tail as u32;
        for level in (0..tail).rev() {
            let end = if level == 0 {
                layout.slice_size
            } else {
                layout.mips[level - 1].offset
            };
            if there(layout.mips[level].offset, end) {
                first = level as u32;
            } else {
                break;
            }
        }
        Some(first)
    } else {
        let mut first = mips as u32;
        for level in (0..mips).rev() {
            let end = if level == 0 {
                layout.slice_size
            } else {
                layout.mips[level - 1].offset
            };
            if there(layout.mips[level].offset, end) {
                first = level as u32;
            } else {
                break;
            }
        }
        (first < mips as u32).then_some(first)
    }
}

/// Builds a `ResourceDesc` whose mip `0` corresponds to title mip `first`.
pub(crate) fn trimmed_desc(
    desc: &crate::resource::ResourceDesc,
    first: u32,
    title_mips: u32,
) -> crate::resource::ResourceDesc {
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_3D: u32 = 4;
    if first == 0 {
        return *desc;
    }
    let mut d = *desc;
    d.width = (d.width >> first).max(1);
    d.height = if d.dimension == TEXTURE_1D {
        1
    } else {
        (d.height >> first).max(1)
    };
    d.depth_or_array_size = if d.dimension == TEXTURE_3D {
        ((u32::from(d.depth_or_array_size) >> first).max(1)) as u16
    } else {
        d.depth_or_array_size
    };
    d.mip_levels = title_mips.saturating_sub(first).max(1) as u16;
    d
}

/// Title-memory byte span needed to read all levels `first..mips` for a
/// single-slice texture (the smallest levels live first in memory, mip `0` last).
/// Multi-slice textures span all slices up to `layout.size`.
pub(crate) fn resident_tiled_size(layout: &victory_formats::texture::Layout, first: u32) -> u64 {
    if first == 0 || layout.slices != 1 || (first as usize) > layout.mips.len() {
        layout.size
    } else {
        layout.mips[first as usize - 1].offset
    }
}

/// Shifts an SRV description authored in title mip space (`0..title_mips`) onto
/// a host resource whose mip `0` is title mip `first` and has `host_mips`
/// levels.
///
/// - When `MostDetailedMip >= first`: `MostDetailedMip -= first`, `MipLevels`
///   clamped to remaining levels, `ResourceMinLODClamp` unchanged (still relative
///   to `MostDetailedMip`).
/// - When `MostDetailedMip < first`: `MostDetailedMip = 0`, `MipLevels` reduced
///   by `first - MostDetailedMip` (clamped to `[1, host_mips]`), and
///   `ResourceMinLODClamp` reduced by the same shift (clamped to `>= 0.0`).
pub(crate) fn shift_srv_desc(
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
    first: u32,
    host_mips: u32,
) {
    if first == 0 || host_mips == 0 {
        return;
    }
    let shift = |most: &mut u32, mips: &mut u32, min_lod: &mut f32| {
        if *most >= first {
            let shifted = *most - first;
            if shifted >= host_mips {
                *most = host_mips - 1;
                if *mips != u32::MAX {
                    *mips = 1;
                }
            } else {
                *most = shifted;
                if *mips != u32::MAX {
                    *mips = (*mips).clamp(1, host_mips - shifted);
                }
            }
        } else {
            let fewer = first - *most;
            *most = 0;
            if *mips != u32::MAX {
                *mips = mips.saturating_sub(fewer).clamp(1, host_mips);
            }
            if *min_lod > 0.0 {
                *min_lod = (*min_lod - fewer as f32).max(0.0);
            }
        }
    };
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE1D => {
            // SAFETY: `ViewDimension` selects `Texture1D`.
            let v = unsafe { &mut desc.Anonymous.Texture1D };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
            // SAFETY: `ViewDimension` selects `Texture1DArray`.
            let v = unsafe { &mut desc.Anonymous.Texture1DArray };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: `ViewDimension` selects `Texture2D`.
            let v = unsafe { &mut desc.Anonymous.Texture2D };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: `ViewDimension` selects `Texture2DArray`.
            let v = unsafe { &mut desc.Anonymous.Texture2DArray };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE3D => {
            // SAFETY: `ViewDimension` selects `Texture3D`.
            let v = unsafe { &mut desc.Anonymous.Texture3D };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURECUBE => {
            // SAFETY: `ViewDimension` selects `TextureCube`.
            let v = unsafe { &mut desc.Anonymous.TextureCube };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
            // SAFETY: `ViewDimension` selects `TextureCubeArray`.
            let v = unsafe { &mut desc.Anonymous.TextureCubeArray };
            shift(&mut v.MostDetailedMip, &mut v.MipLevels, &mut v.ResourceMinLODClamp);
        }
        _ => {}
    }
}

/// Translates a title subresource index to a trimmed host subresource index,
/// returning `None` if the title mip `< first_level` is not resident on the host.
pub(crate) fn map_subresource(
    title_sub: u32,
    first_level: u32,
    title_mips: u32,
    slices: u32,
) -> Option<u32> {
    if first_level == 0 {
        return Some(title_sub);
    }
    if title_mips <= first_level || slices == 0 {
        return None;
    }
    let plane_subs = title_mips.checked_mul(slices)?;
    let plane = title_sub / plane_subs;
    let in_plane = title_sub % plane_subs;
    let slice = in_plane / title_mips;
    let mip = in_plane % title_mips;
    if mip < first_level {
        return None;
    }
    let host_mips = title_mips - first_level;
    Some(plane * slices * host_mips + slice * host_mips + (mip - first_level))
}

/// Lists `(dst_subresource, src_subresource)` pairs for copying shared resident
/// mip levels between two textures of the same title geometry when at least one
/// has `first_level > 0`.
pub(crate) fn copy_resource_subresources(
    dst_first: u32,
    src_first: u32,
    title_mips: u32,
    slices: u32,
    planes: u32,
) -> Vec<(u32, u32)> {
    let common_first = dst_first.max(src_first);
    if common_first >= title_mips || slices == 0 || planes == 0 {
        return Vec::new();
    }
    let dst_host_mips = title_mips - dst_first;
    let src_host_mips = title_mips - src_first;
    let mut pairs =
        Vec::with_capacity((planes * slices * (title_mips - common_first)) as usize);
    for plane in 0..planes {
        for slice in 0..slices {
            for mip in common_first..title_mips {
                let dst_sub =
                    plane * slices * dst_host_mips + slice * dst_host_mips + (mip - dst_first);
                let src_sub =
                    plane * slices * src_host_mips + slice * src_host_mips + (mip - src_first);
                pairs.push((dst_sub, src_sub));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Pipes, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

    #[test]
    fn shift_srv_desc_handles_all_dimensions_and_clamps() {
        // 1. MostDetailedMip < first (0 < 3), MipLevels extends past end, clamp 3.5 -> 0.5.
        let mut srv2d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 12,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 3.5,
                },
            },
        };
        shift_srv_desc(&mut srv2d, 3, 8);
        // SAFETY: `ViewDimension` selects `Texture2D`.
        let v2d = unsafe { srv2d.Anonymous.Texture2D };
        assert_eq!(v2d.MostDetailedMip, 0);
        assert_eq!(v2d.MipLevels, 8);
        assert_eq!(v2d.ResourceMinLODClamp, 0.5);

        // 2. View only asks for unmapped mips (0..2 when first = 3): clamps to host mip 0 (1 level).
        let mut unmapped_only = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 2,
                    FirstArraySlice: 1,
                    ArraySize: 4,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 1.5,
                },
            },
        };
        shift_srv_desc(&mut unmapped_only, 3, 8);
        // SAFETY: `ViewDimension` selects `Texture2DArray`.
        let varr = unsafe { unmapped_only.Anonymous.Texture2DArray };
        assert_eq!(varr.MostDetailedMip, 0);
        assert_eq!(varr.MipLevels, 1);
        assert_eq!(varr.ResourceMinLODClamp, 0.0);
        assert_eq!(varr.FirstArraySlice, 1);
        assert_eq!(varr.ArraySize, 4);

        // 3. MostDetailedMip >= first (5 >= 3): shifts down by 3, keeps clamp relative to view.
        let mut cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV {
                    MostDetailedMip: 5,
                    MipLevels: 4,
                    ResourceMinLODClamp: 1.25,
                },
            },
        };
        shift_srv_desc(&mut cube, 3, 8);
        // SAFETY: `ViewDimension` selects `TextureCube`.
        let vcube = unsafe { cube.Anonymous.TextureCube };
        assert_eq!(vcube.MostDetailedMip, 2);
        assert_eq!(vcube.MipLevels, 4);
        assert_eq!(vcube.ResourceMinLODClamp, 1.25);

        // 4. u32::MAX MipLevels stays u32::MAX across 1D / 3D / CubeArray.
        let mut srv3d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV {
                    MostDetailedMip: 1,
                    MipLevels: u32::MAX,
                    ResourceMinLODClamp: 2.5,
                },
            },
        };
        shift_srv_desc(&mut srv3d, 3, 8);
        // SAFETY: `ViewDimension` selects `Texture3D`.
        let v3d = unsafe { srv3d.Anonymous.Texture3D };
        assert_eq!(v3d.MostDetailedMip, 0);
        assert_eq!(v3d.MipLevels, u32::MAX);
        assert_eq!(v3d.ResourceMinLODClamp, 0.5);
    }

    #[test]
    fn subresource_mapping_skips_unmapped_and_translates_resident() {
        // 5 title mips, 2 array slices, first_level = 2 -> host has 3 mips (title 2, 3, 4).
        assert_eq!(map_subresource(0, 2, 5, 2), None); // slice 0 mip 0
        assert_eq!(map_subresource(1, 2, 5, 2), None); // slice 0 mip 1
        assert_eq!(map_subresource(2, 2, 5, 2), Some(0)); // slice 0 mip 2 -> host 0
        assert_eq!(map_subresource(4, 2, 5, 2), Some(2)); // slice 0 mip 4 -> host 2
        assert_eq!(map_subresource(5, 2, 5, 2), None); // slice 1 mip 0
        assert_eq!(map_subresource(6, 2, 5, 2), None); // slice 1 mip 1
        assert_eq!(map_subresource(7, 2, 5, 2), Some(3)); // slice 1 mip 2 -> host 3
        assert_eq!(map_subresource(9, 2, 5, 2), Some(5)); // slice 1 mip 4 -> host 5

        // Planar (plane 1, slice 0, mip 3): title_sub = 1 * 10 + 3 = 13 -> host = 1 * 6 + 1 = 7.
        assert_eq!(map_subresource(13, 2, 5, 2), Some(7));
        assert_eq!(map_subresource(11, 2, 5, 2), None);

        // Copy between dst (first=1 -> 4 host mips) and src (first=3 -> 2 host mips), 2 slices.
        let pairs = copy_resource_subresources(1, 3, 5, 2, 1);
        assert_eq!(
            pairs,
            vec![
                (2, 0), // slice 0 mip 3
                (3, 1), // slice 0 mip 4
                (6, 2), // slice 1 mip 3
                (7, 3), // slice 1 mip 4
            ]
        );
    }

    #[test]
    fn resident_first_level_ignores_unmapped_levels_and_grows_when_mapped() {
        use windows_sys::Win32::System::Memory::*;
        // 1024x1024 BC7 (11 mips), 64KB S_X: first_tail_mip = 3 (mips 0, 1, 2 outside tail).
        let desc = TextureDesc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth_or_array: 1,
            mips: 11,
            format: 98,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        };
        let layout = Texture::for_gpu(&desc, Pipes::SERIES_X, 0)
            .unwrap()
            .planes
            .into_iter()
            .next()
            .unwrap()
            .layout;
        assert!(layout.first_tail_mip >= 3);
        let total = layout.size as usize;

        // SAFETY: private virtual reservation owned by this test.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), total, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());

            // 1. Nothing committed -> None (nothing sampled from unmapped level).
            assert_eq!(resident_first_level(&layout, base, u64::MAX), None);

            // 2. Commit tail + mip (first_tail_mip - 1) and mip 0 (leaving a gap at mip 1!).
            // Must stop before the gap at mip 1 so unmapped mip 1 is never included.
            let tail = layout.first_tail_mip as usize;
            let commit_end = layout.mips[0].offset as usize; // everything except mip 0
            let _ = commit_end;
            // Commit only [0 .. mips[tail - 2].offset) (tail + mip tail-1).
            let resident_prefix = layout.mips[tail - 2].offset as usize;
            assert_ne!(
                VirtualAlloc(base.cast(), resident_prefix, MEM_COMMIT, PAGE_READWRITE),
                std::ptr::null_mut()
            );
            // Also commit mip 0 alone while mip 1..tail-2 stay uncommitted!
            let mip0_off = layout.mips[0].offset as usize;
            assert_ne!(
                VirtualAlloc(
                    base.add(mip0_off).cast(),
                    total - mip0_off,
                    MEM_COMMIT,
                    PAGE_READWRITE
                ),
                std::ptr::null_mut()
            );
            assert_eq!(
                resident_first_level(&layout, base, u64::MAX),
                Some((tail - 1) as u32)
            );

            // 3. Commit the remaining middle mips -> grows all the way to level 0.
            assert_ne!(
                VirtualAlloc(
                    base.add(resident_prefix).cast(),
                    mip0_off - resident_prefix,
                    MEM_COMMIT,
                    PAGE_READWRITE
                ),
                std::ptr::null_mut()
            );
            assert_eq!(resident_first_level(&layout, base, u64::MAX), Some(0));

            // 4. Overlapping neighbor limit truncates mip 0 -> falls back to level 1.
            assert_eq!(
                resident_first_level(&layout, base, layout.mips[0].offset),
                Some(1)
            );

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn growth_swaps_view_uploads_and_samples_only_resident_levels() {
        use std::sync::Arc;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        use windows_sys::Win32::System::Memory::*;
        use windows::core::{Interface, s};

        let Some(host) = crate::host::host() else {
            return;
        };

        struct ResetForce;
        impl Drop for ResetForce {
            fn drop(&mut self) {
                FORCE.set(false);
            }
        }
        FORCE.set(true);
        let _reset = ResetForce;

        // 256x256 RGBA8 (format 28), 6 mips, SW_64KB_S_X:
        // mip 0 = 256x256, mip 1 = 128x128, mips 2..6 (64x64..8x8) sit in the tail [0..mips[1].offset).
        const SIDE: u32 = 256;
        const MIPS: u32 = 6;
        let td = TextureDesc {
            dimension: Dimension::Tex2d,
            width: SIDE,
            height: SIDE,
            depth_or_array: 1,
            mips: MIPS,
            format: 28,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        };
        let layout = Texture::for_gpu(&td, Pipes::SERIES_X, 0)
            .unwrap()
            .planes
            .into_iter()
            .next()
            .unwrap()
            .layout;
        assert_eq!(layout.first_tail_mip, 2);
        let tail_end = layout.mips[1].offset as usize;
        let mip0_off = layout.mips[0].offset as usize;
        let total = layout.size as usize;

        // SAFETY: private virtual reservation owned by this test.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), total, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>()
        };
        assert!(!base.is_null());

        struct FreeMem(*mut u8);
        impl Drop for FreeMem {
            fn drop(&mut self) {
                // SAFETY: `self.0` came from `VirtualAlloc(..., MEM_RESERVE, ...)`.
                unsafe {
                    let _ = VirtualFree(self.0.cast(), 0, MEM_RELEASE);
                }
            }
        }
        let _free_mem = FreeMem(base);

        // 1. Commit only the tail (mips 2..6); mips 0 and 1 remain PAGE_NOACCESS.
        unsafe {
            assert_ne!(
                VirtualAlloc(base.cast(), tail_end, MEM_COMMIT, PAGE_READWRITE),
                std::ptr::null_mut()
            );
            let tail_slice = std::slice::from_raw_parts_mut(base, tail_end);
            let fill = |buf: &mut [u8], mip: u32, rgba: [u8; 4]| {
                let w = (SIDE >> mip) as usize;
                let h = (SIDE >> mip) as usize;
                let mut lin = vec![0u8; w * h * 4];
                for px in lin.chunks_exact_mut(4) {
                    px.copy_from_slice(&rgba);
                }
                layout.tile(buf, mip, 0, &lin, w * 4, lin.len()).unwrap();
            };
            fill(tail_slice, 2, [0x44, 0x10, 0x20, 0xff]);
            fill(tail_slice, 3, [0x88, 0x10, 0x20, 0xff]);
        }

        // 2. Create placed title texture & mirror slot 1 in a registered HostHeap.
        let resource_words: [u32; 14] = [
            3,
            0,
            0,
            0,
            SIDE,
            0,
            SIDE,
            (MIPS << 16) | 1,
            28,
            1,
            0,
            0x100 | 25,
            0,
            0,
        ];
        let tex = crate::resource::texture_at(base as u64, &resource_words).unwrap();
        let mut srv_words = [0u32; 16];
        srv_words[0] = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
        srv_words[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
        srv_words[2] = 5768;
        srv_words[4] = 0; // MostDetailedMip = 0 (title space)
        srv_words[5] = MIPS; // MipLevels = 6
        srv_words[10] = base as u32;
        srv_words[11] = (base as u64 >> 32) as u32;

        let mut records = [0u64; 4];
        let handle = records.as_mut_ptr() as u64;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource_words, &srv_words, handle);

        let heap = Arc::new(crate::descriptor::host::HostHeap::new(false, 8).unwrap());
        crate::descriptor::host::register_view_heap(&heap);
        assert!(heap.mirror(1, handle, true).is_some());

        // Host copy starts at mip 2 (64x64, 4 mips) because mips 0..2 are unmapped.
        assert_eq!(crate::resource::host_first_level(tex), 2);
        let first_copy = crate::resource::host_copy(tex).unwrap();
        // SAFETY: live resource.
        let first_desc = unsafe { first_copy.texture.GetDesc() };
        assert_eq!(first_desc.Width, 64);
        assert_eq!(first_desc.Height, 64);
        assert_eq!(first_desc.MipLevels, 4);
        assert_eq!(heap.shows(1, &first_copy.texture), Some(true));

        // 3. Upload initial resident levels using a HostList + queue, and copy into
        // a full destination texture via copy_resource (must skip mips 0..2 cleanly).
        let dst_mem = vec![0u8; total];
        let dst_tex = crate::resource::texture_at(dst_mem.as_ptr() as u64, &resource_words).unwrap();
        let queue_iid = crate::object::guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let alloc_iid = crate::object::guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = crate::object::guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut alloc, mut list) = (std::ptr::null(), std::ptr::null(), std::ptr::null());
        let qdesc = [0u32; 4];
        assert_eq!(
            crate::queue::create(std::ptr::null(), qdesc.as_ptr().cast(), &queue_iid, &mut queue),
            crate::object::S_OK
        );
        assert_eq!(
            crate::list::create_allocator(std::ptr::null(), 0, &alloc_iid, &mut alloc),
            crate::object::S_OK
        );
        assert_eq!(
            crate::list::create_list(std::ptr::null(), 0, 0, alloc, std::ptr::null(), &list_iid, &mut list),
            crate::object::S_OK
        );
        let host_list = crate::list::host_list(list).unwrap();
        crate::texture::upload(host_list, tex, &first_copy.texture);
        // Copy trimmed `tex` -> full `dst_tex` (translating subresources & skipping unmapped).
        unsafe {
            let vt = *list.cast::<*const *const usize>();
            let copy_res: extern "system" fn(crate::object::This, crate::object::This, crate::object::This) =
                std::mem::transmute(*vt.add(17));
            copy_res(list, dst_tex, tex);
            crate::list::ensure_state(host_list, tex, 0x40);
            let close: extern "system" fn(crate::object::This) -> crate::object::HRESULT =
                std::mem::transmute(*vt.add(9));
            assert_eq!(close(list), crate::object::S_OK);
            let qvt = *queue.cast::<*const *const usize>();
            let exec: extern "system" fn(crate::object::This, u32, *const crate::object::This) =
                std::mem::transmute(*qvt.add(10));
            exec(queue, 1, &list);
        }
        let sync = host.new_list().unwrap();
        sync.close();
        host.execute(&[&sync]);

        // Sample slot 1 via compute shader: mip 0 samples largest resident level (mip 2 = 0x44),
        // mip 1 samples title mip 3 (0x88).
        const SAMPLE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors = 1)), UAV(u0), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT)"
Texture2D<float4> tex : register(t0);
SamplerState nearest : register(s0);
RWByteAddressBuffer out_words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void sample_mips() {
    float4 m0 = tex.SampleLevel(nearest, float2(0.5, 0.5), 0);
    float4 m1 = tex.SampleLevel(nearest, float2(0.5, 0.5), 1);
    float4 m2 = tex.SampleLevel(nearest, float2(0.5, 0.5), 2);
    out_words.Store(0, (uint)(m0.x * 255.0 + 0.5));
    out_words.Store(4, (uint)(m1.x * 255.0 + 0.5));
    out_words.Store(8, (uint)(m2.x * 255.0 + 0.5));
}
"#;
        let run_sample = |heap: &crate::descriptor::host::HostHeap| -> [u32; 3] {
            // SAFETY: valid D3D12 compute dispatch on `host.device`.
            unsafe {
                let code = crate::host::compile(SAMPLE, s!("sample_mips"), s!("cs_5_1")).unwrap();
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
                let pso: ID3D12PipelineState = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                let uav_buf = host.uav_buffer(256).unwrap();
                let readback = host.read_back_buffer(256).unwrap();
                let dqueue: ID3D12CommandQueue = host
                    .device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                let dalloc: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let dlist: ID3D12GraphicsCommandList = host
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &dalloc, Some(&pso))
                    .unwrap();
                dlist.SetComputeRootSignature(&root);
                dlist.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                dlist.SetComputeRootDescriptorTable(0, heap.gpu_handle(1));
                dlist.SetComputeRootUnorderedAccessView(1, uav_buf.GetGPUVirtualAddress());
                dlist.Dispatch(1, 1, 1);
                dlist.ResourceBarrier(&[crate::host::transition(
                    &uav_buf,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                dlist.CopyResource(&readback, &uav_buf);
                dlist.Close().unwrap();
                dqueue.ExecuteCommandLists(&[Some(dlist.cast().unwrap())]);
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                dqueue.Signal(&fence, 1).unwrap();
                let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(1, ev).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let mut ptr = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let words = std::slice::from_raw_parts(ptr.cast::<u32>(), 3);
                let res = [words[0], words[1], words[2]];
                readback.Unmap(0, None);
                res
            }
        };

        let sampled_trimmed = run_sample(&heap);
        assert_eq!(sampled_trimmed[0], 0x44, "mip 0 clamps to largest resident level (mip 2)");
        assert_eq!(sampled_trimmed[1], 0x88, "mip 1 reads resident title mip 3");

        // 4. Now commit mips 1 and 0, fill mip 0 with 0xee and mip 1 with 0xaa,
        // notify via remapped + texture_slots_remapped, grow host copy, upload,
        // swap slot 1's view, and retire the old copy.
        unsafe {
            assert_ne!(
                VirtualAlloc(
                    base.add(tail_end).cast(),
                    total - tail_end,
                    MEM_COMMIT,
                    PAGE_READWRITE
                ),
                std::ptr::null_mut()
            );
            let full_slice = std::slice::from_raw_parts_mut(base, total);
            let fill = |buf: &mut [u8], mip: u32, rgba: [u8; 4]| {
                let w = (SIDE >> mip) as usize;
                let h = (SIDE >> mip) as usize;
                let mut lin = vec![0u8; w * h * 4];
                for px in lin.chunks_exact_mut(4) {
                    px.copy_from_slice(&rgba);
                }
                layout.tile(buf, mip, 0, &lin, w * 4, lin.len()).unwrap();
            };
            fill(full_slice, 1, [0xaa, 0x10, 0x20, 0xff]);
            fill(full_slice, 0, [0xee, 0x10, 0x20, 0xff]);
        }
        let _ = mip0_off;
        crate::descriptor::host::remapped(base as usize, total);
        assert!(heap.texture_slots_remapped(10) >= 1);

        let sync = crate::resource::sync_resident_levels(tex, true);
        let crate::resource::ResidentSync::Swapped {
            new_host,
            old_host,
            old_raw,
        } = sync
        else {
            panic!("expected host copy growth after mapping mips 0 and 1");
        };
        assert_eq!(new_host.first_level, 0);
        assert_ne!(new_host.id, old_host.id);
        // SAFETY: live resource.
        let grown_desc = unsafe { new_host.texture.GetDesc() };
        assert_eq!(grown_desc.Width, 256);
        assert_eq!(grown_desc.Height, 256);
        assert_eq!(grown_desc.MipLevels, 6);

        // Reset allocator & list, upload into `new_host`, swap descriptor slot 1, retire `old_host`.
        unsafe {
            let avt = *alloc.cast::<*const *const usize>();
            let areset: extern "system" fn(crate::object::This) -> crate::object::HRESULT =
                std::mem::transmute(*avt.add(4));
            assert_eq!(areset(alloc), crate::object::S_OK);
            let lvt = *list.cast::<*const *const usize>();
            let lreset: extern "system" fn(
                crate::object::This,
                crate::object::This,
                crate::object::This,
            ) -> crate::object::HRESULT = std::mem::transmute(*lvt.add(10));
            assert_eq!(lreset(list, alloc, std::ptr::null()), crate::object::S_OK);
            let host_list = crate::list::host_list(list).unwrap();
            crate::texture::upload(host_list, tex, &new_host.texture);
            crate::list::ensure_state(host_list, tex, 0x40);
            assert!(crate::descriptor::host::swap_texture_views(old_raw) >= 1);
            crate::host::retire(old_host);
            let close: extern "system" fn(crate::object::This) -> crate::object::HRESULT =
                std::mem::transmute(*lvt.add(9));
            assert_eq!(close(list), crate::object::S_OK);
            let qvt = *queue.cast::<*const *const usize>();
            let exec: extern "system" fn(crate::object::This, u32, *const crate::object::This) =
                std::mem::transmute(*qvt.add(10));
            exec(queue, 1, &list);
        }
        let sync2 = host.new_list().unwrap();
        sync2.close();
        host.execute(&[&sync2]);

        assert_eq!(heap.shows(1, &new_host.texture), Some(true));
        let sampled_grown = run_sample(&heap);
        assert_eq!(sampled_grown, [0xee, 0xaa, 0x44]);

        crate::texture::unstage(tex);
        crate::texture::unstage(dst_tex);
        unsafe {
            let release = |obj: crate::object::This| {
                let vt = *obj.cast::<*const *const usize>();
                let rel: extern "system" fn(crate::object::This) -> u32 =
                    std::mem::transmute(*vt.add(2));
                rel(obj);
            };
            release(list);
            release(alloc);
            release(queue);
        }
    }
}
