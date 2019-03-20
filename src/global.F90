module global

  use, intrinsic :: ISO_C_BINDING

#ifdef MPIF08
  use mpi_f08
#endif

  use bank_header,      only: Bank
  use cmfd_header
  use constants
  use dict_header,      only: DictCharInt, DictIntInt
  use geometry_header,  only: Cell, Universe, Lattice, LatticeContainer
  use material_header,  only: Material
  use mesh_header,      only: RegularMesh
  use mgxs_header,      only: Mgxs, MgxsContainer
  use nuclide_header
  use plot_header,      only: ObjectPlot
  use sab_header,       only: SAlphaBeta
  use set_header,       only: SetInt
  use stl_vector,       only: VectorInt
  use surface_header,   only: SurfaceContainer
  use source_header,    only: SourceDistribution
  use tally_header,     only: TallyObject, TallyDerivative
  use tally_filter_header, only: TallyFilterContainer, TallyFilterMatch
  use trigger_header,   only: KTrigger
  use timer_header,     only: Timer
  use volume_header,    only: VolumeCalculation

  implicit none

  ! ============================================================================
  ! GEOMETRY-RELATED VARIABLES

  ! These dictionaries provide a fast lookup mechanism -- the key is the
  ! user-specified identifier and the value is the index in the corresponding
  ! array

  ! Number of lost particles
  integer :: n_lost_particles

  ! ============================================================================
  ! MULTI-GROUP CROSS SECTION RELATED VARIABLES

  ! Cross section arrays
  type(MgxsContainer), allocatable, target :: nuclides_MG_(:)

  ! ============================================================================
  ! TALLY-RELATED VARIABLES

  type(TallyFilterMatch), allocatable :: filter_matches(:)

  ! Pointers for different tallies
  type(TallyObject), pointer :: user_tallies(:) => null()

  ! Starting index (minus 1) in tallies for each tally group
  integer :: i_user_tallies = -1
  integer :: i_cmfd_tallies = -1

  ! Global tallies
  !   1) collision estimate of k-eff
  !   2) absorption estimate of k-eff
  !   3) track-length estimate of k-eff
  !   4) leakage fraction

  ! It is possible to protect accumulate operations on global tallies by using
  ! an atomic update. However, when multiple threads accumulate to the same
  ! global tally, it can cause a higher cache miss rate due to
  ! invalidation. Thus, we use threadprivate variables to accumulate global
  ! tallies and then reduce at the end of a generation.
  real(8) :: global_tally_leakage     = ZERO
!$omp threadprivate(global_tally_leakage)

  integer :: n_meshes       = 0 ! # of structured meshes
  integer :: n_user_meshes  = 0 ! # of structured user meshes
  integer :: n_filters      = 0 ! # of filters
  integer :: n_user_filters = 0 ! # of user filters
  integer :: n_user_tallies = 0 ! # of user tallies

  ! Normalization for statistics
  integer :: n_realizations = 0 ! # of independent realizations

  ! Flag for turning tallies on
  logical :: tallies_on = .false.
  logical :: active_batches = .false.

  ! ============================================================================
  ! TALLY PRECISION TRIGGER VARIABLES

  logical :: satisfy_triggers = .false.       ! whether triggers are satisfied

  ! Source and fission bank
  type(Bank), allocatable, target :: source_bank(:)
#ifdef _OPENMP
  type(Bank), allocatable, target :: master_fission_bank(:)
#endif
  integer(8) :: work         ! number of particles per processor
  integer(8), allocatable :: work_index(:) ! starting index in source bank for each process
  integer(8) :: current_work ! index in source bank of current history simulated

  ! Temporary k-effective values
  real(8) :: keff_std         ! standard deviation of average k
  real(8) :: k_col_abs = ZERO ! sum over batches of k_collision * k_absorption
  real(8) :: k_col_tra = ZERO ! sum over batches of k_collision * k_tracklength
  real(8) :: k_abs_tra = ZERO ! sum over batches of k_absorption * k_tracklength

  type(RegularMesh), pointer :: entropy_mesh

  type(RegularMesh), pointer :: ufs_mesh => null()

  ! ============================================================================
  ! PARALLEL PROCESSING VARIABLES

#ifdef _OPENMP
  integer :: thread_id             ! ID of a given thread
#endif

  ! ============================================================================
  ! TIMING VARIABLES

  type(Timer) :: time_total         ! timer for total run
  type(Timer) :: time_initialize    ! timer for initialization
  type(Timer) :: time_unionize      ! timer for material xs-energy grid union
  type(Timer) :: time_bank          ! timer for fission bank synchronization
  type(Timer) :: time_bank_sample   ! timer for fission bank sampling
  type(Timer) :: time_bank_sendrecv ! timer for fission bank SEND/RECV
  type(Timer) :: time_tallies       ! timer for accumulate tallies
  type(Timer) :: time_inactive      ! timer for inactive batches
  type(Timer) :: time_active        ! timer for active batches
  type(Timer) :: time_transport     ! timer for transport only
  type(Timer) :: time_finalize      ! timer for finalization

  ! ============================================================================
  ! MISCELLANEOUS VARIABLES

  ! Eigenvalue
  integer    :: current_batch     ! current batch
  integer    :: current_gen       ! current generation within a batch

  ! Restart run
  integer :: restart_batch

  ! Frequency mesh
  logical                    :: flux_frequency_on = .false.
  logical                    :: precursor_frequency_on = .false.
  real(8), allocatable       :: flux_frequency(:) ! Omega by energy group
  real(8), allocatable       :: precursor_frequency(:,:) ! Omega by cell and delayed group
  type(RegularMesh), pointer :: frequency_mesh
  integer                    :: num_frequency_energy_groups = ZERO
  integer                    :: num_frequency_delayed_groups = ZERO
  real(8), allocatable       :: frequency_energy_bins(:)
  real(8), allocatable       :: frequency_energy_bin_avg(:)


contains

!===============================================================================
! FREE_MEMORY deallocates and clears  all global allocatable arrays in the
! program
!===============================================================================

  subroutine free_memory()

    integer :: i ! Loop Index

    ! Deallocate cross section data, listings, and cache
    if (allocated(nuclides)) then
    ! First call the clear routines
      do i = 1, size(nuclides)
        call nuclides(i) % clear()
      end do
      deallocate(nuclides)
    end if
    if (allocated(libraries)) deallocate(libraries)

    if (allocated(res_scat_nuclides)) deallocate(res_scat_nuclides)

    if (allocated(nuclides_MG_)) deallocate(nuclides_MG_)

    ! Deallocate fission and source bank and entropy
!$omp parallel
    if (allocated(filter_matches)) deallocate(filter_matches)
!$omp end parallel
#ifdef _OPENMP
    if (allocated(master_fission_bank)) deallocate(master_fission_bank)
#endif
    if (allocated(source_bank)) deallocate(source_bank)

    ! Deallocate array of work indices
    if (allocated(work_index)) deallocate(work_index)

    ! Deallocate CMFD
    call deallocate_cmfd(cmfd)

    ! Deallocate track_identifiers
    if (allocated(track_identifiers)) deallocate(track_identifiers)

    ! Deallocate dictionaries
    call nuclide_dict % clear()
    call library_dict % clear()

    ! Clear statepoint and sourcepoint batch set
    call statepoint_batch % clear()
    call sourcepoint_batch % clear()

    ! Deallocate entropy mesh
    if (associated(entropy_mesh)) then
      if (allocated(entropy_mesh % lower_left)) &
           deallocate(entropy_mesh % lower_left)
      if (allocated(entropy_mesh % upper_right)) &
           deallocate(entropy_mesh % upper_right)
      if (allocated(entropy_mesh % width)) deallocate(entropy_mesh % width)
      deallocate(entropy_mesh)
    end if

    ! Deallocate ufs
    if (associated(ufs_mesh)) then
        if (allocated(ufs_mesh % lower_left)) deallocate(ufs_mesh % lower_left)
        if (allocated(ufs_mesh % upper_right)) &
             deallocate(ufs_mesh % upper_right)
        if (allocated(ufs_mesh % width)) deallocate(ufs_mesh % width)
        deallocate(ufs_mesh)
    end if

  end subroutine free_memory

!===============================================================================
! OVERALL_GENERATION determines the overall generation number
!===============================================================================

  pure function overall_generation() result(gen)
    integer :: gen
    gen = gen_per_batch*(current_batch - 1) + current_gen
  end function overall_generation

end module global
