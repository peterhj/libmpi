extern crate libc;

use libc::{c_void, c_char, c_int, c_long, c_longlong};
use std::mem::{transmute};

#[repr(C)]
pub enum MPI_Error {
  Success       = 0,
  Buffer        = 1,
  Count         = 2,
  Type          = 3,
  Tag           = 4,
  Comm          = 5,
  Rank          = 6,
  RequestOrRoot = 7,
  Group         = 8,
  Op            = 9,
  Topology      = 10,
  Dims          = 11,
  Arg           = 12,
  Unknown       = 13,
  Truncate      = 14,
  Other         = 15,
  Internal      = 16,
  InStatus      = 17,
  Pending       = 18,
  Access        = 19,
  Amode         = 20,
  Assert        = 21,
  BadFile       = 22,
  Base          = 23,
  Conversion    = 24,
  Disp          = 25,
  DupDatarep    = 26,
  FileExists    = 27,
  FileInuse     = 28,
  File          = 29,
  InfoKey       = 30,
  InfoNokey     = 31,
  InfoValue     = 32,
  Info          = 33,
  Io            = 34,
  Keyval        = 35,
  Locktype      = 36,
  Name          = 37,
  NoMem         = 38,
  NotSame       = 39,
  NoSpace       = 40,
  NoSuchFile    = 41,
  Port          = 42,
  Quota         = 43,
  ReadOnly      = 44,
  RmaConflict   = 45,
  RmaSync       = 46,
  Service       = 47,
  Size          = 48,
  Spawn         = 49,
  UnsupDatarep  = 50,
  UnsupOp       = 51,
  Window        = 52,
  LastCode      = 53,
  OutOfResource = -2,
}

pub const MPI_THREAD_SINGLE:      c_int = 0;
pub const MPI_THREAD_FUNNELED:    c_int = 1;
pub const MPI_THREAD_SERIALIZED:  c_int = 2;
pub const MPI_THREAD_MULTIPLE:    c_int = 3;

pub const MPI_ANY_SOURCE: c_int = -2;
pub const MPI_ANY_TAG:    c_int = -1;

pub const MPI_LOCK_EXCLUSIVE: c_int = 234;
pub const MPI_LOCK_SHARED:    c_int = 235;

pub const MPI_MAX_PORT_NAME:  c_int = 256;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Count(pub c_longlong);

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Offset(pub c_longlong);

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Aint(pub c_long);

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Status {
  count_lo:     c_int,
  count_hi_and_cancelled: c_int,
  pub source:   c_int,
  pub tag:      c_int,
  pub error:    c_int,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Datatype(pub c_int);

/*pub const MPI_CHAR:                 MPI_Datatype = MPI_Datatype(0x4c000101);
pub const MPI_SIGNED_CHAR:          MPI_Datatype = MPI_Datatype(0x4c000118);
pub const MPI_UNSIGNED_CHAR:        MPI_Datatype = MPI_Datatype(0x4c000102);
pub const MPI_BYTE:                 MPI_Datatype = MPI_Datatype(0x4c00010d);
pub const MPI_LONG:                 MPI_Datatype = MPI_Datatype(0x4c000807);
pub const MPI_UNSIGNED_LONG:        MPI_Datatype = MPI_Datatype(0x4c000808);
pub const MPI_LONG_LONG_INT:        MPI_Datatype = MPI_Datatype(0x4c000809);
pub const MPI_UNSIGNED_LONG_LONG:   MPI_Datatype = MPI_Datatype(0x4c000819);
pub const MPI_FLOAT:                MPI_Datatype = MPI_Datatype(0x4c00040a);
pub const MPI_DOUBLE:               MPI_Datatype = MPI_Datatype(0x4c00080b);*/

/*pub static MPI_BYTE:    *const ompi_datatype_t = &ompi_mpi_byte as *const _;
pub static MPI_UNSIGNED_LONG_LONG: *const ompi_datatype_t = &ompi_mpi_unsigned_long_long as *const _;
pub static MPI_FLOAT:   *const ompi_datatype_t = &ompi_mpi_float as *const _;
pub static MPI_DOUBLE:  *const ompi_datatype_t = &ompi_mpi_double as *const _;*/

impl MPI_Datatype {
  pub fn UNSIGNED_CHAR() -> MPI_Datatype {
    MPI_Datatype(0x4c000102)
  }

  pub fn BYTE() -> MPI_Datatype {
    MPI_Datatype(0x4c00010d)
  }

  pub fn UNSIGNED_LONG_LONG() -> MPI_Datatype {
    MPI_Datatype(0x4c000819)
  }

  pub fn FLOAT() -> MPI_Datatype {
    MPI_Datatype(0x4c00040a)
  }

  pub fn DOUBLE() -> MPI_Datatype {
    MPI_Datatype(0x4c00080b)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Comm(pub c_int);

/*pub const MPI_COMM_WORLD:   MPI_Comm = MPI_Comm(0x44000000);
pub const MPI_COMM_SELF:    MPI_Comm = MPI_Comm(0x44000001);*/

//pub static MPI_COMM_WORLD: *const ompi_communicator_t = &ompi_mpi_comm_world as *const _;

impl MPI_Comm {
  pub fn NULL() -> MPI_Comm {
    MPI_Comm(0x04000000)
  }

  pub fn SELF() -> MPI_Comm {
    MPI_Comm(0x44000001)
  }

  pub fn WORLD() -> MPI_Comm {
    MPI_Comm(0x44000000)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Group(pub c_int);

//pub const MPI_GROUP_EMPTY:  MPI_Group = MPI_Group(0x48000000);

impl MPI_Group {
  pub fn NULL() -> MPI_Group {
    MPI_Group(0x08000000)
  }

  pub fn EMPTY() -> MPI_Group {
    MPI_Group(0x48000000)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Info(pub c_int);

impl MPI_Info {
  pub fn NULL() -> MPI_Info {
    MPI_Info(0x1c000000)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Op(pub c_int);

/*pub const MPI_MAX:          MPI_Op = MPI_Op(0x58000001);
pub const MPI_MIN:          MPI_Op = MPI_Op(0x58000002);
pub const MPI_SUM:          MPI_Op = MPI_Op(0x58000003);
pub const MPI_REPLACE:      MPI_Op = MPI_Op(0x5800000d);
pub const MPI_NO_OP:        MPI_Op = MPI_Op(0x5800000e);*/

//pub static MPI_SUM: *const ompi_op_t = &ompi_mpi_op_sum as *const _;

impl MPI_Op {
  pub fn SUM() -> MPI_Op {
    MPI_Op(0x58000003)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Request(pub c_int);

impl MPI_Request {
  pub fn NULL() -> MPI_Request {
    MPI_Request(0x2c000000)
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Win(pub c_int);

impl MPI_Win {
  pub fn NULL() -> MPI_Win {
    MPI_Win(0x20000000)
  }
}

//pub const MPI_WIN_NULL:     MPI_Win = MPI_Win(0x20000000);

/*#[link(name = "dl")]
extern "C" {
}

#[link(name = "hwloc")]
extern "C" {
}*/

// FIXME(20161127): linking with Intel MPI.
#[link(name = "mpicxx")]
extern "C" {}
#[link(name = "mpifort")]
extern "C" {}
#[link(name = "mpi")]
extern "C" {}
#[link(name = "mpigi", kind = "static")]
extern "C" {}
#[link(name = "dl")]
extern "C" {}
#[link(name = "rt")]
extern "C" {}
#[link(name = "pthread")]
extern "C" {}

//#[link(name = "mpich")]
extern "C" {
  /*pub fn MPI_Init(argc: *mut c_int, argv: *mut *mut *mut c_char) -> c_int;
  pub fn MPI_Initialized() -> c_int;
  pub fn MPI_Finalize() -> c_int;
  pub fn MPI_Abort(comm: MPI_Comm, errorcode: c_int) -> c_int;

  pub fn MPI_Comm_size(comm: MPI_Comm, size: *mut c_int) -> c_int;
  pub fn MPI_Comm_rank(comm: MPI_Comm, rank: *mut c_int) -> c_int;

  pub fn MPI_Send(buf: *const c_void, count: c_int, datatype: MPI_Datatype, dest: c_int, tag: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Recv(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, source: c_int, tag: c_int, comm: MPI_Comm, status: *mut MPI_Status) -> c_int;

  pub fn MPI_Barrier(comm: MPI_Comm) -> c_int;
  pub fn MPI_Bcast(buf: *const c_void, count: c_int, datatype: MPI_Datatype, root: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Allreduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> c_int;

  pub fn MPI_Info_create(info: *mut MPI_Info) -> c_int;

  pub fn MPI_Win_create(base: *mut c_void, size: MPI_Aint, disp_unit: c_int, info: MPI_Info, comm: MPI_Comm, win: *mut MPI_Win) -> c_int;
  pub fn MPI_Win_free(win: *mut MPI_Win) -> c_int;
  pub fn MPI_Win_fence(assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_lock(lock_type: c_int, rank: c_int, assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_unlock(rank: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Get(origin_addr: *mut c_void, origin_count: c_int, origin_datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, target_count: c_int, target_datatype: MPI_Datatype, win: MPI_Win) -> c_int;*/

  pub fn MPI_Init(argc: *mut c_int, argv: *mut *mut *mut c_char) -> c_int;
  pub fn MPI_Init_thread(argc: *mut c_int, argv: *mut *mut *mut c_char, required: c_int, provided: *mut c_int) -> c_int;
  pub fn MPI_Initialized(flag: *mut c_int) -> c_int;
  pub fn MPI_Finalize() -> c_int;
  pub fn MPI_Abort(comm: MPI_Comm, errorcode: c_int) -> c_int;

  pub fn MPI_Alloc_mem(size: MPI_Aint, info: MPI_Info, baseptr: *mut *mut c_void) -> c_int;
  pub fn MPI_Free_mem(base: *mut c_void) -> c_int;

  pub fn MPI_Comm_size(comm: MPI_Comm, size: *mut c_int) -> c_int;
  pub fn MPI_Comm_rank(comm: MPI_Comm, rank: *mut c_int) -> c_int;
  pub fn MPI_Comm_group(comm: MPI_Comm, group: *mut MPI_Group) -> c_int;
  pub fn MPI_Comm_create(comm: MPI_Comm, group: MPI_Group, newcomm: *mut MPI_Comm) -> c_int;

  pub fn MPI_Send(buf: *const c_void, count: c_int, datatype: MPI_Datatype, dest: c_int, tag: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Probe(source: c_int, tag: c_int, comm: MPI_Comm, status: *mut MPI_Status) -> c_int;
  pub fn MPI_Recv(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, source: c_int, tag: c_int, comm: MPI_Comm, status: *mut MPI_Status) -> c_int;
  pub fn MPI_Sendrecv(
      sendbuf: *const c_void, sendcount: c_int, sendtype: MPI_Datatype, dest: c_int, sendtag: c_int,
      recvbuf: *mut c_void, recvcount: c_int, recvtype: MPI_Datatype, source: c_int, recvtag: c_int,
      comm: MPI_Comm, status: *mut MPI_Status,
  ) -> c_int;

  pub fn MPI_Isend(buf: *const c_void, count: c_int, datatype: MPI_Datatype, dest: c_int, tag: c_int, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;
  pub fn MPI_Issend(buf: *const c_void, count: c_int, datatype: MPI_Datatype, dest: c_int, tag: c_int, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;
  pub fn MPI_Iprobe(source: c_int, tag: c_int, comm: MPI_Comm, flag: *mut c_int, status: *mut MPI_Status) -> c_int;
  pub fn MPI_Irecv(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, source: c_int, tag: c_int, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;
  pub fn MPI_Test(request: *mut MPI_Request, flag: *mut c_int, status: *mut MPI_Status) -> c_int;
  pub fn MPI_Wait(request: *mut MPI_Request, status: *mut MPI_Status) -> c_int;
  pub fn MPI_Waitall(count: c_int, requests: *mut MPI_Request, statuses: *mut MPI_Status) -> c_int;

  pub fn MPI_Barrier(comm: MPI_Comm) -> c_int;
  pub fn MPI_Bcast(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, root: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Allreduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> c_int;

  pub fn MPI_Ibcast(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, root: c_int, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;
  pub fn MPI_Ireduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: MPI_Datatype, op: MPI_Op, root: c_int, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;
  pub fn MPI_Iallreduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm, request: *mut MPI_Request) -> c_int;

  pub fn MPI_Open_port(info: MPI_Info, port_name: *mut c_char) -> c_int;
  pub fn MPI_Publish_name(service_name: *const c_char, info: MPI_Info, port_name: *const c_char) -> c_int;
  pub fn MPI_Lookup_name(service_name: *const c_char, info: MPI_Info, port_name: *mut c_char) -> c_int;
  pub fn MPI_Comm_accept(port_name: *const c_char, info: MPI_Info, root: c_int, comm: MPI_Comm, newcomm: *mut MPI_Comm) -> c_int;
  pub fn MPI_Comm_connect(port_name: *const c_char, info: MPI_Info, root: c_int, comm: MPI_Comm, newcomm: *mut MPI_Comm) -> c_int;

  pub fn MPI_Group_range_excl(group: MPI_Group, n: c_int, ranges: *mut c_int, newgroup: *mut MPI_Group) -> c_int;
  pub fn MPI_Group_range_incl(group: MPI_Group, n: c_int, ranges: *mut c_int, newgroup: *mut MPI_Group) -> c_int;

  pub fn MPI_Info_create(info: *mut MPI_Info) -> c_int;
  pub fn MPI_Info_set(info: MPI_Info, key: *const c_char, value: *const c_char) -> c_int;

  pub fn MPI_Win_create(base: *mut c_void, size: MPI_Aint, disp_unit: c_int, info: MPI_Info, comm: MPI_Comm, win: *mut MPI_Win) -> c_int;
  pub fn MPI_Win_free(win: *mut MPI_Win) -> c_int;
  pub fn MPI_Win_fence(assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_lock(lock_type: c_int, rank: c_int, assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_unlock(rank: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_start(group: MPI_Group, assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_complete(win: MPI_Win) -> c_int;
  pub fn MPI_Win_post(group: MPI_Group, assert: c_int, win: MPI_Win) -> c_int;
  pub fn MPI_Win_wait(win: MPI_Win) -> c_int;
  pub fn MPI_Win_test(win: MPI_Win, flag: *mut c_int) -> c_int;
  pub fn MPI_Compare_and_swap(origin_addr: *const c_void, compare_addr: *const c_void, result_addr: *mut c_void, datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, win: MPI_Win) -> c_int;
  pub fn MPI_Get(origin_addr: *mut c_void, origin_count: c_int, origin_datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, target_count: c_int, target_datatype: MPI_Datatype, win: MPI_Win) -> c_int;
  /*pub fn MPI_Get_accumulate(origin_addr: *mut c_void, origin_count: c_int, origin_datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, target_count: c_int, target_datatype: MPI_Datatype, op: MPI_Op,  win: MPI_Win) -> c_int;*/
  pub fn MPI_Put(origin_addr: *const c_void, origin_count: c_int, origin_datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, target_count: c_int, target_datatype: MPI_Datatype, win: MPI_Win) -> c_int;
  pub fn MPI_Accumulate(origin_addr: *const c_void, origin_count: c_int, origin_datatype: MPI_Datatype, target_rank: c_int, target_disp: MPI_Aint, target_count: c_int, target_datatype: MPI_Datatype, op: MPI_Op, win: MPI_Win) -> c_int;
}
