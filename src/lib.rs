#![feature(optin_builtin_traits)]

#[cfg(feature = "mpich")]
extern crate mpich as mpi_ffi;
#[cfg(feature = "openmpi")]
extern crate openmpi as mpi_ffi;

#[macro_use]
extern crate lazy_static;
extern crate libc;

use mpi_ffi::*;

use libc::{c_void, c_int};
use std::env;
use std::ffi::{CString, CStr};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

type AintTy = isize;

lazy_static! {
  static ref SESSION:   Arc<MpiSession> = {
    Arc::new(MpiSession::new())
  };
}

pub struct MpiSession {
  th_level: AtomicUsize,
}

impl Drop for MpiSession {
  fn drop(&mut self) {
    let is_init = unsafe { MPI_Initialized() };
    if is_init != 0 {
      unsafe { MPI_Finalize() };
    }
  }
}

impl MpiSession {
  pub fn new() -> MpiSession {
    MpiSession{
      th_level: AtomicUsize::new(0),
    }
  }

  pub fn set_thread_level(&self, level: MpiThreadLevel) -> Result<bool, ()> {
    match self.th_level.compare_and_swap(0, level as usize, Ordering::AcqRel) {
      0 => {
        let args: Vec<_> = env::args().collect();
        // FIXME(20160130): this leaks the C string.
        let mut c_args: Vec<_> = args.into_iter().map(|s| match CString::new(s) {
          Ok(s) => s.into_raw(),
          Err(e) => panic!("mpi: failed to initialize: bad argv: {:?}", e),
        }).collect();
        let mut argc = c_args.len() as c_int;
        let mut argv = (&mut c_args).as_mut_ptr();
        //unsafe { MPI_Init(&mut argc as *mut _, &mut argv as *mut _) };
        let mut provided: c_int = -1;
        match level {
          MpiThreadLevel::Single => {
            unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_SINGLE,     &mut provided as *mut _) };
            assert_eq!(provided, MPI_THREAD_SINGLE);
          }
          MpiThreadLevel::Funneled => {
            unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_FUNNELED,   &mut provided as *mut _) };
            assert_eq!(provided, MPI_THREAD_FUNNELED);
          }
          MpiThreadLevel::Serialized => {
            unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_SERIALIZED, &mut provided as *mut _) };
            assert_eq!(provided, MPI_THREAD_SERIALIZED);
          }
          MpiThreadLevel::Multithreaded => {
            unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_MULTIPLE,   &mut provided as *mut _) };
            assert_eq!(provided, MPI_THREAD_MULTIPLE);
          }
        }
        Ok(true)
      }
      x => {
        if x == level as usize {
          let is_init = unsafe { MPI_Initialized() };
          assert!(is_init != 0);
          Ok(false)
        } else {
          panic!("tried to set MPI thread level to {:?} ({}), but already set to {}", level, level as usize, x);
        }
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MpiThreadLevel {
  Single            = 1,
  Funneled          = 2,
  Serialized        = 3,
  Multithreaded     = 4,
}

#[derive(Clone)]
pub struct MpiCtx {
  session:  Arc<MpiSession>,
}

impl MpiCtx {
  pub fn new(level: MpiThreadLevel) -> MpiCtx {
    let session = SESSION.clone();
    match session.set_thread_level(level) {
      Ok(_) => {}
      Err(_) => unimplemented!(),
    }
    MpiCtx{session: session}
  }
}

pub trait MpiData {
  fn datatype() -> MPI_Datatype;
}

impl MpiData for u8 {
  fn datatype() -> MPI_Datatype {
    MPI_Datatype::BYTE()
  }
}

impl MpiData for u64 {
  fn datatype() -> MPI_Datatype {
    MPI_Datatype::UNSIGNED_LONG_LONG()
  }
}

impl MpiData for f32 {
  fn datatype() -> MPI_Datatype {
    MPI_Datatype::FLOAT()
  }
}

impl MpiData for f64 {
  fn datatype() -> MPI_Datatype {
    MPI_Datatype::DOUBLE()
  }
}

pub trait MpiOp {
  fn op() -> MPI_Op;
}

pub struct MpiSumOp;

impl MpiOp for MpiSumOp {
  fn op() -> MPI_Op {
    MPI_Op::SUM()
  }
}

pub struct MpiStatus {
  pub src_rank: usize,
  pub tag:      c_int,
  pub error:    c_int,
}

impl MpiStatus {
  pub fn new(ffi_status: MPI_Status) -> MpiStatus {
    MpiStatus{
      src_rank: ffi_status.source as usize,
      tag:      ffi_status.tag,
      error:    ffi_status.error,
    }
  }
}

pub struct MpiRequest {
  inner:    MPI_Request,
}

/*impl MpiRequest {
  pub fn nonblocking_send<T>(buf: &[T], dst: usize, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Isend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }

  pub fn nonblocking_sync_send<T>(buf: &[T], dst: usize, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Issend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }

  pub fn nonblocking_recv<T>(buf: &mut [T], src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }
}*/

impl MpiRequest {
  pub fn query(&mut self) -> Result<Option<MpiStatus>, c_int> {
    // FIXME(20160416)
    unimplemented!();
  }

  pub fn wait(&mut self) -> Result<MpiStatus, c_int> {
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Wait(&mut self.inner as *mut _, &mut status as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiStatus{
      src_rank: status.source as usize,
      tag:      status.tag,
      error:    status.error,
    })
  }
}

pub struct MpiRequestList {
  reqs:   Vec<MPI_Request>,
  stats:  Vec<MPI_Status>,
}

impl MpiRequestList {
  pub fn new() -> MpiRequestList {
    MpiRequestList{
      reqs:   vec![],
      stats:  vec![],
    }
  }

  pub fn clear(&mut self) {
    self.reqs.clear();
    self.stats.clear();
  }

  pub fn append(&mut self, request: MpiRequest) {
    self.reqs.push(request.inner);
    self.stats.push(MPI_Status::default());
  }

  pub fn wait_all(&mut self) -> Result<(), c_int> {
    if self.reqs.is_empty() {
      return Ok(());
    }
    let code = unsafe { MPI_Waitall(self.reqs.len() as c_int, self.reqs.as_mut_ptr(), self.stats.as_mut_ptr()) };
    if code != 0 {
      return Err(code);
    }
    self.reqs.clear();
    self.stats.clear();
    Ok(())
  }
}

pub struct MpiMemory<T> {
  base: *mut T,
  len:  usize,
}

unsafe impl<T> Send for MpiMemory<T> where T: Send {}

impl<T> Drop for MpiMemory<T> {
  fn drop(&mut self) {
    let code = unsafe { MPI_Free_mem(self.base as *mut _) };
    if code != 0 {
      panic!("MPI_Free_mem failed: {}", code);
    }
  }
}

impl<T> MpiMemory<T> {
  pub fn alloc(len: usize) -> Result<MpiMemory<T>, c_int> {
    let mut base = null_mut();
    let code = unsafe { MPI_Alloc_mem(MPI_Aint((size_of::<T>() * len) as AintTy), MPI_Info::NULL(), &mut base as *mut *mut T as *mut *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiMemory{
      base: base,
      len:  len,
    })
  }
}

impl<T> AsRef<[T]> for MpiMemory<T> {
  fn as_ref(&self) -> &[T] {
    unsafe { from_raw_parts(self.base as *const T, self.len) }
  }
}

impl<T> AsMut<[T]> for MpiMemory<T> {
  fn as_mut(&mut self) -> &mut [T] {
    unsafe { from_raw_parts_mut(self.base, self.len) }
  }
}

pub struct MpiComm {
  inner:    MPI_Comm,
  predef:   bool,
}

impl Drop for MpiComm {
  fn drop(&mut self) {
    if !self.predef {
      // FIXME(20160419)
      //unsafe { MPI_Comm_disconnect(&mut self.inner as *mut _) };
    }
  }
}

impl MpiComm {
  pub fn self_(_: &MpiCtx) -> MpiComm {
    MpiComm{
      inner:    MPI_Comm::SELF(),
      predef:   true,
    }
  }

  pub fn world(_: &MpiCtx) -> MpiComm {
    MpiComm{
      inner:    MPI_Comm::WORLD(),
      predef:   true,
    }
  }

  pub fn rank(&self) -> Result<usize, c_int> {
    let mut rank = 0;
    let code = unsafe { MPI_Comm_rank(self.inner, &mut rank as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(rank as usize)
  }

  pub fn size(&self) -> Result<usize, c_int> {
    let mut size = 0;
    let code = unsafe { MPI_Comm_size(self.inner, &mut size as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(size as usize)
  }

  pub fn group(&self) -> Result<MpiGroup, c_int> {
    let mut group = MPI_Group::NULL();
    let code = unsafe { MPI_Comm_group(self.inner, &mut group as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiGroup{
      inner:  group,
    })
  }

  pub fn create(&self, group: &MpiGroup) -> Result<MpiComm, c_int> {
    let mut newcomm = MPI_Comm::NULL();
    let code = unsafe { MPI_Comm_create(self.inner, group.inner, &mut newcomm as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiComm{
      inner:    newcomm,
      predef:   false,
    })
  }

  pub fn accept(port_name: &CStr) -> Result<MpiComm, c_int> {
    let mut inner = MPI_Comm::NULL();
    let code = unsafe { MPI_Comm_accept(port_name.as_ptr(), MPI_Info::NULL(), 0, MPI_Comm::SELF(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiComm{
      inner:  inner,
      predef: false,
    })
  }

  pub fn connect(port_name: &CStr) -> Result<MpiComm, c_int> {
    let mut inner = MPI_Comm::NULL();
    let code = unsafe { MPI_Comm_connect(port_name.as_ptr(), MPI_Info::NULL(), 0, MPI_Comm::SELF(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiComm{
      inner:  inner,
      predef: false,
    })
  }

  pub fn barrier(&self) -> Result<(), c_int> {
    let code = unsafe { MPI_Barrier(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn nonblocking_send<T>(&self, buf: &[T], dst: usize, tag: i32) -> Result<MpiRequest, c_int> where T: MpiData {
    let mut request = MPI_Request::NULL();
    let code = unsafe { MPI_Isend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }

  pub fn nonblocking_sync_send<T>(&self, buf: &[T], dst: usize, tag: i32) -> Result<MpiRequest, c_int> where T: MpiData {
    let mut request = MPI_Request::NULL();
    let code = unsafe { MPI_Issend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }

  pub fn nonblocking_probe(&self, src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<Option<MpiStatus>, c_int> {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut flag = 0;
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Iprobe(src_rank, tag, self.inner, &mut flag as *mut _, &mut status as *mut _) };
    if code != 0 {
      return Err(code);
    }
    match flag {
      0 => Ok(None),
      1 => Ok(Some(MpiStatus::new(status))),
      _ => unreachable!(),
    }
  }

  pub fn nonblocking_recv<T>(&self, buf: &mut [T], src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = MPI_Request::NULL();
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
    })
  }

  pub fn nonblocking_broadcast<T>(&self, buf: &mut [T], root: usize) -> Result<MpiRequest, c_int>
  where T: MpiData {
    let mut req = MPI_Request::NULL();
    let code = unsafe { MPI_Ibcast(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), root as c_int, self.inner, &mut req as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: req,
    })
  }

  pub fn nonblocking_reduce<T, Op>(&self, src_buf: &[T], dst_buf: &mut [T], _op: Op, root: usize) -> Result<MpiRequest, c_int>
  where T: MpiData, Op: MpiOp {
    assert_eq!(src_buf.len(), dst_buf.len());
    let mut req = MPI_Request::NULL();
    let code = unsafe { MPI_Ireduce(src_buf.as_ptr() as *const c_void, dst_buf.as_mut_ptr() as *mut c_void, src_buf.len() as c_int, T::datatype(), Op::op(), root as c_int, self.inner, &mut req as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: req,
    })
  }

  pub fn nonblocking_allreduce<T, Op>(&self, src_buf: &[T], dst_buf: &mut [T], _op: Op) -> Result<MpiRequest, c_int>
  where T: MpiData, Op: MpiOp {
    assert_eq!(src_buf.len(), dst_buf.len());
    let mut req = MPI_Request::NULL();
    let code = unsafe { MPI_Iallreduce(src_buf.as_ptr() as *const c_void, dst_buf.as_mut_ptr() as *mut c_void, src_buf.len() as c_int, T::datatype(), Op::op(), self.inner, &mut req as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: req,
    })
  }
}

pub struct MpiGroup {
  inner:    MPI_Group,
}

impl Drop for MpiGroup {
  fn drop(&mut self) {
    // FIXME(201607xx)
  }
}

impl MpiGroup {
  pub fn empty() -> MpiGroup {
    MpiGroup{inner: MPI_Group::EMPTY()}
  }

  pub fn ranges(&self, ranges: &[(usize, usize, usize)]) -> Result<MpiGroup, c_int> {
    let mut c_ranges: Vec<c_int> = Vec::with_capacity(3 * ranges.len());
    for i in 0 .. ranges.len() {
      c_ranges.push(ranges[i].0 as c_int);
      c_ranges.push((ranges[i].1 - 1) as c_int);
      c_ranges.push(ranges[i].2 as c_int);
    }
    let mut new_inner = MPI_Group::NULL();
    let code = unsafe { MPI_Group_range_incl(self.inner, ranges.len() as c_int, c_ranges.as_mut_ptr(), &mut new_inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiGroup{inner: new_inner})
  }
}

pub struct MpiInfo {
  inner:    MPI_Info,
}

impl Drop for MpiInfo {
  fn drop(&mut self) {
    // FIXME(201607xx)
  }
}

impl MpiInfo {
  pub fn null() -> MpiInfo {
    MpiInfo{
      inner:    MPI_Info::NULL(),
    }
  }

  /*pub fn create(/*_mpi: &Mpi*/) -> Result<MpiInfo, c_int> {
    let mut inner = MPI_Info::NULL();
    let code = unsafe { MPI_Info_create(&mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiInfo{
      inner:    inner,
    })
  }*/

  pub fn set(&mut self, key: &CStr, value: &CStr) -> Result<(), c_int> {
    let code = unsafe { MPI_Info_set(self.inner, key.as_ptr(), value.as_ptr()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

#[derive(Clone, Copy)]
pub enum MpiWindowFenceFlag {
  Null,
}

#[derive(Clone, Copy)]
pub enum MpiWindowLockMode {
  Exclusive,
  Shared,
}

pub trait MpiWindowStorage<T> {
  fn storage_ptr(&self) -> *const T;
  fn storage_len(&self) -> usize;
}

pub trait MpiWindowMutStorage<T>: MpiWindowStorage<T> {
  fn storage_mut_ptr(&mut self) -> *mut T;
}

impl<T> MpiWindowStorage<T> for Vec<T> {
  fn storage_ptr(&self) -> *const T {
    self.as_ptr()
  }

  fn storage_len(&self) -> usize {
    self.len()
  }
}

impl<T> MpiWindowMutStorage<T> for Vec<T> {
  fn storage_mut_ptr(&mut self) -> *mut T {
    self.as_mut_ptr()
  }
}

impl<T> MpiWindowStorage<T> for Mutex<Vec<T>> {
  fn storage_ptr(&self) -> *const T {
    self.lock().unwrap().as_ptr()
  }

  fn storage_len(&self) -> usize {
    self.lock().unwrap().len()
  }
}

impl<T> MpiWindowMutStorage<T> for Mutex<Vec<T>> {
  fn storage_mut_ptr(&mut self) -> *mut T {
    self.lock().unwrap().as_mut_ptr()
  }
}

pub struct MpiOwnedWindow<T, Storage> {
  buf:      Storage,
  inner:    MPI_Win,
  _marker:  PhantomData<T>,
}

unsafe impl<T, Storage> Send for MpiOwnedWindow<T, Storage> where T: Send, Storage: Send {}
unsafe impl<T, Storage> Sync for MpiOwnedWindow<T, Storage> where T: Send, Storage: Send {}

impl<T, Storage> Drop for MpiOwnedWindow<T, Storage> {
  fn drop(&mut self) {
    // FIXME(20160415): need to do a fence before freeing, otherwise it will
    // cause a seg fault!
    unsafe { MPI_Win_fence(0, self.inner) };
    unsafe { MPI_Win_free(&mut self.inner as *mut _) };
  }
}

impl<T, Storage> MpiOwnedWindow<T, Storage> where Storage: AsRef<[T]> + AsMut<[T]> {
  pub fn as_slice(&self) -> &[T] {
    self.buf.as_ref()
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.buf.as_mut()
  }
}

impl<T, Storage> MpiOwnedWindow<T, Storage> where Storage: MpiWindowMutStorage<T> {
  pub fn create(mut buf: Storage) -> Result<MpiOwnedWindow<T, Storage>, c_int> {
    let mut inner = MPI_Win::NULL();
    let code = unsafe { MPI_Win_create(buf.storage_mut_ptr() as *mut _, MPI_Aint((size_of::<T>() * buf.storage_len()) as AintTy), size_of::<T>() as c_int, MPI_Info::NULL(), MPI_Comm::WORLD(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiOwnedWindow{
      buf:      buf,
      inner:    inner,
      _marker:  PhantomData,
    })
  }

  pub fn storage(&self) -> &Storage {
    &self.buf
  }

  pub fn storage_mut(&mut self) -> &mut Storage {
    &mut self.buf
  }
}

impl<T, Storage> MpiOwnedWindow<T, Storage> {
  pub fn fence(&self, /*_flag: MpiOwnedWindowFenceFlag*/) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_fence(assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn lock(&self, target_rank: usize, mode: MpiWindowLockMode) -> Result<(), c_int> {
    let lock_type = match mode {
      MpiWindowLockMode::Exclusive => MPI_LOCK_EXCLUSIVE,
      MpiWindowLockMode::Shared => MPI_LOCK_SHARED,
    };
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_lock(lock_type, target_rank as c_int, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn unlock(&self, target_rank: usize) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_unlock(target_rank as c_int, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn start(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 1; // MPI_MODE_NOCHECK
    let code = unsafe { MPI_Win_start(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn complete(&self) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_complete(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn post(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_post(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn wait(&self) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_wait(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn test(&self) -> Result<bool, c_int> {
    let mut flag = 0;
    let code = unsafe { MPI_Win_test(self.inner, &mut flag as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(flag != 0)
  }
}

impl<T, Storage> MpiOwnedWindow<T, Storage> where T: MpiData, Storage: MpiWindowStorage<T> {
  pub unsafe fn compare_and_swap(&self, origin_addr: *const T, compare_addr: *const T, result_addr: *mut T, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let buf_len = self.buf.storage_len();
    //assert!(buf_len >= ); // FIXME(201607xx)
    let code = MPI_Compare_and_swap(
        origin_addr as *const _,
        compare_addr as *const _,
        result_addr as *mut _,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn get(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let buf_len = self.buf.storage_len();
    assert!(origin_len <= buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let buf_len = self.buf.storage_len();
    assert!(origin_len <= buf_len);
    let code = MPI_Put(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn accumulate<Op>(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize, _op: Op) -> Result<(), c_int>
  where Op: MpiOp {
    let buf_len = self.buf.storage_len();
    assert!(origin_len <= buf_len);
    let code = MPI_Accumulate(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        Op::op(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

pub struct MpiUnsafeWindow<T> {
  buf_ptr:  *mut T,
  buf_len:  usize,
  inner:    MPI_Win,
}

unsafe impl<T> Send for MpiUnsafeWindow<T> where T: Send {}

impl<T> Drop for MpiUnsafeWindow<T> {
  fn drop(&mut self) {
    // FIXME(20160415): need to do a fence before freeing, otherwise it will
    // cause a seg fault!
    unsafe { MPI_Win_fence(0, self.inner) };
    unsafe { MPI_Win_free(&mut self.inner as *mut _) };
  }
}

impl<T> MpiUnsafeWindow<T> {
  pub unsafe fn create(buf_ptr: *mut T, buf_len: usize) -> Result<MpiUnsafeWindow<T>, c_int> {
    let mut inner = MPI_Win::NULL();
    let code = unsafe { MPI_Win_create(buf_ptr as *mut _, MPI_Aint((size_of::<T>() * buf_len) as AintTy), size_of::<T>() as c_int, MPI_Info::NULL(), MPI_Comm::WORLD(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiUnsafeWindow{
      buf_ptr:  buf_ptr,
      buf_len:  buf_len,
      inner:    inner,
    })
  }

  pub unsafe fn as_slice(&self) -> &[T] {
    //self.buf.as_ref()
    from_raw_parts(self.buf_ptr as *const T, self.buf_len)
  }

  pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
    //self.buf.as_mut()
    from_raw_parts_mut(self.buf_ptr, self.buf_len)
  }

  pub fn fence(&self, /*_flag: MpiUnsafeWindowFenceFlag*/) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_fence(assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn lock(&self, target_rank: usize, mode: MpiWindowLockMode) -> Result<(), c_int> {
    let lock_type = match mode {
      MpiWindowLockMode::Exclusive => MPI_LOCK_EXCLUSIVE,
      MpiWindowLockMode::Shared => MPI_LOCK_SHARED,
    };
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_lock(lock_type, target_rank as c_int, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn unlock(&self, target_rank: usize) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_unlock(target_rank as c_int, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn start(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 1; // MPI_MODE_NOCHECK
    let code = unsafe { MPI_Win_start(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn complete(&self) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_complete(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn post(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_post(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn wait(&self) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_wait(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

impl<T> MpiUnsafeWindow<T> where T: MpiData {
  pub unsafe fn compare_and_swap(&self, origin_addr: *const T, compare_addr: *const T, result_addr: *mut T, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let code = MPI_Compare_and_swap(
        origin_addr as *const _,
        compare_addr as *const _,
        result_addr as *mut _,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn get(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Put(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn accumulate<Op>(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize, _op: Op) -> Result<(), c_int>
  where Op: MpiOp {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Accumulate(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        Op::op(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}
