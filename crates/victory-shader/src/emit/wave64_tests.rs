//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    lowered_for_dispatch(native, p, wide, host_wave, false)
}

fn lowered_for_dispatch(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
    partial: bool,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower_with_thread_limit(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    }, partial);
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        written_root_uavs: lowered.written_root_uavs,
        writes_guest_memory: lowered.writes_guest_memory,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn partial_thread_dispatch_masks_the_final_groups_memory_writes() {
    // Each active thread increments counter[local X]. With two groups the
    // tail's first N lanes report two increments, the others only one.
    let native = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7e04_0281, // v_mov_b32 v2, 1
        0xe0c8_1000, 0x8002_0201, // buffer_atomic_add v2, v1, s[8:11], 0 offen
    ]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let shader = lowered_for_dispatch(&native, &p, &HashSet::new(), 64, true).unwrap();
    let limit = shader.user_data.iter().find(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS_X)).unwrap();
    assert_eq!((limit.slot, limit.dword), (0, 64));
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else { return };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    for count in [0u32, 1, 7, 63, 64, 65, 74, 77, 127, 128] {
        let mut constants = vec![0; shader.user_data_dwords() as usize];
        constants[limit.index as usize] = count;
        let result = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(constants))],
            views: &[View::RawUav(68)],
            samplers: &[],
            groups: [2, 1, 1],
        }).unwrap();
        let active = if count == 0 { 128 } else { count };
        for (i, &word) in result[0].iter().enumerate() {
            let expected = if i < 64 {
                u32::from((i as u32) < active) + u32::from((i as u32 + 64) < active)
            } else { 0 };
            assert_eq!(word, expected, "count {count}, word {i}");
        }
    }
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// A 64-lane compute wave. A lane mask holding lanes outside `EXEC` (saved before a
/// per-lane branch and restored inside it) is later parked in two neighbouring lanes of a
/// VGPR (`v_writelane_b32` / `v_readlane_b32`), while an unrelated 32-bit scalar reuses the
/// lower lane later in the program (`water_shading:CS_TransmissionVolumetrics`,
/// `water_shading:CS_TransmissionVolumetricsSubmerged`). With `VICTORY_UNSPILL_MIXED_PAIRS=0`
/// the pair was left as 32-bit moves and lost its mask tracking; pairing matching saves and
/// saves and restores as `s_mov_b64` keeps the mask tracked through the spill while leaving
/// the 32-bit spill on the same slot.
#[test]
fn parked_mask_pair_sharing_a_lane_with_a_scalar_spill() {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            ir::TEST_UNSPILL_MIXED_PAIRS.set(self.0);
        }
    }
    let _reset = Reset(ir::TEST_UNSPILL_MIXED_PAIRS.replace(Some(false)));
    let words = [
        0x7D88_0000, // 0: v_cmp_gt_u32 vcc, s0, v0
        0xBE94_046A, // 1: s_mov_b64 s[20:21], vcc
        0xBE8C_047E, // 2: s_mov_b64 s[12:13], exec
        0xBF06_8001, // 3: s_cmp_eq_u32 s1, 0
        0xBF85_0007, // 4: s_cbranch_scc1 -> join (12)
        0xBE8E_0414, // 5: s_mov_b64 s[14:15], s[20:21]
        0x7D88_0090, // 6: v_cmp_gt_u32 vcc, 16, v0
        0xBE8C_246A, // 7: s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // 8: s_cbranch_execz -> join (12)
        0xBE94_0480, // 9: s_mov_b64 s[20:21], 0
        0xBE94_040E, // 10: s_mov_b64 s[20:21], s[14:15]
        0xBF80_0000, // 11: s_nop 0
        0xBEFE_040C, // 12: join: s_mov_b64 exec, s[12:13]
        0xD761_0005,
        0x0001_0414, // 13: v_writelane_b32 v5, s20, 2
        0x7E04_0280, // 15: v_mov_b32 v2, 0
        0xD761_0005,
        0x0001_0615, // 16: v_writelane_b32 v5, s21, 3
        0xBE94_0480, // 18: s_mov_b64 s[20:21], 0
        0xD760_0014,
        0x0001_0505, // 19: v_readlane_b32 s20, v5, 2
        0xD760_0015,
        0x0001_0705, // 21: v_readlane_b32 s21, v5, 3
        0xD761_0005,
        0x0001_0401, // 23: v_writelane_b32 v5, s1, 2
        0xD760_0016,
        0x0001_0505, // 25: v_readlane_b32 s22, v5, 2
        0xD501_0001,
        0x0051_0280, // 27: v_cndmask_b32 v1, 0, 1, s[20:21]
        0x4A02_0216, // 29: v_add_nc_u32 v1, s22, v1
        0x7DA4_0094, // 30: v_cmpx_eq_u32 exec, 20, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // 31: buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native(&words);
    let p_off = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build off");
    let err = lowered_on(&native, &p_off, &HashSet::new(), 64);
    assert!(
        matches!(&err, Err(ShaderError::Unsupported { what, .. }) if *what == "a select on an untracked mask"),
        "{:?}",
        err.err()
    );
    ir::TEST_UNSPILL_MIXED_PAIRS.set(Some(true));
    let p_on = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build on");
    let insts: Vec<&str> = p_on.blocks.iter().flat_map(|b| &b.insts).map(|i| i.name()).collect();
    assert!(!insts.contains(&"v_writelane_b32") && !insts.contains(&"v_readlane_b32"));
    assert_eq!(insts.iter().filter(|&&n| n == "s_mov_b64").count(), 9);
    assert_eq!(insts.iter().filter(|&&n| n == "s_mov_b32").count(), 3); // s5 + 2 mixed scalar spills
    let shader = lowered_on(&native, &p_on, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, s1, expected) in [(32u32, 0u32, 1u32), (16, 0, 0), (32, 3, 4), (16, 3, 3)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "lane 20 with {below} lanes in mask, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: mixed-width unspill preserves parked mask pair");
}

/// A PHI joining a leftover lane mask on one arm with an opaque scalar (such as
/// a buffer load) on the other (`shadowmask:CS_TraceShadowRays_FullAdaptive`,
/// `0x87f4..0xae80`) feeds scalar bitwise packing (`s_and_b32`/`s_or_b32`) and a
/// vector compare inside a divergent block. With `VICTORY_MIXED_MASK_PHI=0`
/// the PHI is treated as derived and `s_or_b32` fails as an active-only mask;
/// enabled, the PHI resolves numerically and downstream bitwise ops emit as data.
#[test]
fn mixed_mask_and_opaque_scalar_phi_lowers_downstream_bitwise_ops_as_data() {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            structure::TEST_MIXED_MASK_PHI.set(self.0);
        }
    }
    let _reset = Reset(structure::TEST_MIXED_MASK_PHI.replace(Some(false)));
    // s0: input base, s1: branch selector (0 = scalar s0+3, else = 12-lane mask).
    // On the scalar path s20 = s0 + 3 (opaque scalar); on the mask path
    // s20 = (v0 < 12) (lane mask 0xFFF). After the join, under a divergent EXEC
    // (v0 < 16), s22 = s20 & 7, s23 = s22 | (s0 + 8), compared against v0 and
    // stored from lane 0.
    let words = [
        0x8115_8800, //  0: s_add_i32 s21, s0, 8
        0xBF06_8001, //  1: s_cmp_eq_u32 s1, 0
        0xBF84_0002, //  2: s_cbranch_scc0 -> mask_arm (5)
        0x8114_8300, //  3: scalar_arm: s_add_i32 s20, s0, 3
        0xBF82_0002, //  4: s_branch -> join (7)
        0x7D88_008C, //  5: mask_arm: v_cmp_gt_u32 vcc_lo, 12, v0
        0xBE94_036A, //  6: s_mov_b32 s20, vcc_lo
        0x7E02_0280, //  7: join: v_mov_b32 v1, 0
        0x7D88_0090, //  8: v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE82_3C6A, //  9: s_and_saveexec_b32 s2, vcc_lo
        0xBF88_0006, // 10: s_cbranch_execz -> restore (17)
        0x8716_8714, // 11: s_and_b32 s22, s20, 7
        0x8817_1516, // 12: s_or_b32 s23, s22, s21
        0x7D84_0017, // 13: v_cmp_eq_u32 vcc_lo, s23, v0
        0x7E04_0281, // 14: v_mov_b32 v2, 1
        0x0204_0480, // 15: v_cndmask_b32 v2, 0, v2, vcc_lo
        0x4A02_0417, // 16: v_add_nc_u32 v1, s23, v2
        0xBEFE_0302, // 17: restore: s_mov_b32 exec_lo, s2
        0x7DA4_0080, // 18: v_cmpx_eq_u32 exec_lo, 0, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // 19: buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let err = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&err, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask holding lanes elsewhere read as data"),
        "{:?}",
        err.err()
    );

    structure::TEST_MIXED_MASK_PHI.set(Some(true));
    let st = structure::structure(&p).expect("structure");
    let s_or_val = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|i| i.name() == "s_or_b32")
        .and_then(|i| i.write(ir::Reg::S(23)))
        .expect("s_or_b32 s23");
    assert!(!st.masks.contains_key(&s_or_val), "s_or_b32 s23 must not be tracked as a lane mask");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, s1, expected) in [(0u32, 0u32, 11u32), (0, 1, 15)] {
        let Some(got) = on_the_card(&shader, [s0, s1]) else { return };
        assert_eq!(got, expected, "s0={s0}, s1={s1}");
    }
    eprintln!("ran on the hardware device: mixed mask and opaque scalar PHI keeps bitwise ops as data");
}

#[test]
fn mesh_wave64_first_lane_reads_lane_zero_for_upper_half() {
    use super::container::SigElement;
    use super::host::Draw;
    use crate::native::{MeshIo, PsInput, Usage, VsOutput};
    let mut pos_el = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(1),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    let tc_el = SigElement {
        name: "TEXCOORD".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    let words: [u32; _] = [
        0xD765_0000,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD766_0000,
        0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x4A02_00FF,
        100,         // v_add_nc_u32 v1, 100, v0
        0x7E04_0501, // v_readfirstlane_b32 s2, v1
        0xBE84_047E, // s_mov_b64 s[4:5], exec
        0x7DA4_00A0, // v_cmpx_eq_u32 exec, 32, v0
        0x7E04_0202, // v_mov_b32 v2, s2
        0x7E06_0280, // v_mov_b32 v3, 0
        0xD834_0000,
        0x0000_0203, // ds_write_b32 v3, v2
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xBF8A_0000, // s_barrier
        0xD8D8_0000,
        0x0200_0003, // ds_read_b32 v2, v3
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E08_02FF,
        0xBF80_0000, // v_mov_b32 v4, -1.0
        0x7E0A_02FF,
        0xBF80_0000, // v_mov_b32 v5, -1.0
        0x7E10_02FF,
        0x4040_0000, // v_mov_b32 v8, 3.0
        0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
        0x0208_1104, // v_cndmask_b32 v4, v4, v8
        0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
        0x020A_1105, // v_cndmask_b32 v5, v5, v8
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0xF800_00CF,
        0x0706_0504, // exp pos0, v4, v5, v6, v7
        0xF800_020F,
        0x0702_0202, // exp param0, v2, v2, v2, v7
        0x7E10_02FF,
        0x0020_0400, // v_mov_b32 v8, (0 | (1 << 10) | (2 << 20))
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0xF800_0941,
        0x0000_0008, // exp prim, v8, off, off, off done
        0xBF81_0000, // s_endpgm
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 16).expect("ir::build");
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 0,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            clip_cntl: 0,
            threads: 64,
            max_vertices: 3,
            max_primitives: 1,
            topology: 2,
            lds_bytes: 4,
        }),
        code,
    };
    let osg1 = super::container::signature(&[tc_el.clone(), pos_el.clone()], false);
    let dxbc_ms = crate::dxbc::build(&[(crate::FourCC::OSG1, &osg1)]);
    let c_ms = crate::Container::parse(&dxbc_ms).unwrap();
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("raise");
    let (ms, _) = super::emit_program(&c_ms, &native, &p, &st, &io, &res).expect("emit ms");

    // Simple pixel shader copying TEXCOORD0 to SV_Target0
    pos_el.usage = 0;
    let ps_words: [u32; _] = [
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0x7E06_02F2, // v_mov_b32 v3, 1.0
        0xF800_180F,
        0x0302_0202, // exp mrt0, v2, v2, v2, v3 done vm
        0xBF81_0000,
    ];
    let ps_code: Vec<u8> = ps_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let ps_p = ir::build(&ps_code, false, 8).unwrap();
    let ps_nat = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b1111,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 2,
            db_shader_control: 0,
            patches: vec![],
        },
        code: ps_code,
    };
    let mut target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 15,
        half: false,
    };
    target.usage = 15;
    let isg1 = super::container::signature(&[tc_el], true);
    let osg1_ps = super::container::signature(&[target], false);
    let dxbc_ps = crate::dxbc::build(&[(crate::FourCC::ISG1, &isg1), (crate::FourCC::OSG1, &osg1_ps)]);
    let c_ps = crate::Container::parse(&dxbc_ps).unwrap();
    let ps_st = structure::structure(&ps_p).unwrap();
    let ps_res = resource::track(&ps_p, &ps_nat.layout, ps_nat.stage, Some(&ps_st)).unwrap();
    let ps_io = io::raise(&ps_p, &ps_nat).unwrap();
    let (ps, _) = super::emit_program(&c_ps, &ps_nat, &ps_p, &ps_st, &ps_io, &ps_res).unwrap();

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut ms.dxil.clone()).expect("ms DXIL validation failed");
        v.validate(&mut ps.dxil.clone()).expect("ps DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = host.mesh_pipeline(&ms, &ps).expect("mesh_pipeline");
    let img = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: vec![],
                textures: &[],
                samplers: &[],
                vertices: &[],
                stride: 0,
                size: (4, 4),
                mesh_groups: 1,
            },
        )
        .expect("draw mesh");
    eprintln!("mesh_wave64 first_lane from thread 32 = {:?}", img[0]);
    assert_eq!(img[0][0], 100.0, "upper half read lane 32 ({}) instead of lane 0 (100.0)", img[0][0]);
}
