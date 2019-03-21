module mesh

  use algorithm,  only: binary_search
  use bank_header, only: bank
  use constants
  use mesh_header
  use message_passing

  implicit none

contains

!===============================================================================
! GET_MESH_BIN determines the tally bin for a particle in a structured mesh
!===============================================================================

  pure subroutine get_mesh_bin(m, xyz, bin)
    type(RegularMesh), intent(in) :: m      ! mesh pointer
    real(8), intent(in)           :: xyz(:) ! coordinates
    integer, intent(out)          :: bin    ! tally bin

    integer :: n       ! size of mesh
    integer :: d       ! mesh dimension index
    integer :: ijk(3)  ! indices in mesh
    logical :: in_mesh ! was given coordinate in mesh at all?

    ! Get number of dimensions
    n = m % n_dimension

    ! Loop over the dimensions of the mesh
    do d = 1, n

      ! Check for cases where particle is outside of mesh
      if (xyz(d) < m % lower_left(d) - TINY_BIT) then
        bin = NO_BIN_FOUND
        return
      elseif (xyz(d) > m % upper_right(d) + TINY_BIT) then
        bin = NO_BIN_FOUND
        return
      end if
    end do

    ! Determine indices
    call get_mesh_indices(m, xyz, ijk, in_mesh)

    ! Convert indices to bin
    if (in_mesh) then
      bin = mesh_indices_to_bin(m, ijk)
    else
      bin = NO_BIN_FOUND
    end if

  end subroutine get_mesh_bin

!===============================================================================
! GET_MESH_INDICES determines the indices of a particle in a structured mesh
!===============================================================================

  pure subroutine get_mesh_indices(m, xyz, ijk, in_mesh)
    type(RegularMesh), intent(in) :: m
    real(8), intent(in)           :: xyz(:)  ! coordinates to check
    integer, intent(out)          :: ijk(:)  ! indices in mesh
    logical, intent(out)          :: in_mesh ! were given coords in mesh?
    integer                       :: n       ! size of mesh
    integer                       :: d       ! mesh dimension index

    ! Get number of dimensions
    n = m % n_dimension

    ! Find particle in mesh
    ijk(:n) = ceiling((xyz(:n) - m % lower_left)/m % width)

    ! Loop over the dimensions of the mesh
    do d = 1, n

      ! Put points in the boundary in the mesh cells
      if (ijk(d) == 0 .and. xyz(d) >= m % lower_left(d) - TINY_BIT) then
        ijk(d) = 1
      else if (ijk(d) == m % dimension(d) + 1 .and. xyz(d) <= m % upper_right(d) + TINY_BIT) then
        ijk(d) = m % dimension(d)
      end if
    end do

    ! Determine if particle is in mesh
    if (any(ijk(:n) < 1) .or. any(ijk(:n) > m % dimension)) then
      in_mesh = .false.
    else
      in_mesh = .true.
    end if

  end subroutine get_mesh_indices

!===============================================================================
! MESH_INDICES_TO_BIN maps (i), (i,j), or (i,j,k) indices to a single bin number
! for use in a TallyObject results array
!===============================================================================

  pure function mesh_indices_to_bin(m, ijk) result(bin)
    type(RegularMesh), intent(in) :: m
    integer, intent(in)           :: ijk(:)
    integer                       :: bin

    if (m % n_dimension == 1) then
      bin = ijk(1)
    elseif (m % n_dimension == 2) then
      bin = (ijk(2) - 1) * m % dimension(1) + ijk(1)
    elseif (m % n_dimension == 3) then
      bin = ((ijk(3) - 1) * m % dimension(2) + (ijk(2) - 1)) &
           * m % dimension(1) + ijk(1)
    end if

  end function mesh_indices_to_bin

!===============================================================================
! BIN_TO_MESH_INDICES maps a single mesh bin from a TallyObject results array to
! (i), (i,j), or (i,j,k) indices
!===============================================================================

  pure subroutine bin_to_mesh_indices(m, bin, ijk)
    type(RegularMesh), intent(in) :: m
    integer, intent(in)           :: bin
    integer, intent(out)          :: ijk(:)

    if (m % n_dimension == 1) then
      ijk(1) = bin
    else if (m % n_dimension == 2) then
      ijk(1) = mod(bin - 1, m % dimension(1)) + 1
      ijk(2) = (bin - 1)/m % dimension(1) + 1
    else if (m % n_dimension == 3) then
      ijk(1) = mod(bin - 1, m % dimension(1)) + 1
      ijk(2) = mod(bin - 1, m % dimension(1) * m % dimension(2)) &
           / m % dimension(1) + 1
      ijk(3) = (bin - 1)/(m % dimension(1) * m % dimension(2)) + 1
    end if

  end subroutine bin_to_mesh_indices


!===============================================================================
! COUNT_BANK_SITES determines the number of fission bank sites in each cell of a
! given mesh as well as an optional energy group structure. This can be used for
! a variety of purposes (Shannon entropy, CMFD, uniform fission source
! weighting)
!===============================================================================

  subroutine count_bank_sites(m, bank_array, cnt, energies, size_bank, &
       sites_outside)

    type(RegularMesh), intent(in) :: m             ! mesh to count sites
    type(Bank), intent(in)     :: bank_array(:) ! fission or source bank
    real(8),    intent(out)    :: cnt(:,:)      ! weight of sites in each
    ! cell and energy group
    real(8), intent(in),    optional :: energies(:)   ! energy grid to search
    integer(8), intent(in), optional :: size_bank     ! # of bank sites (on each proc)
    logical, intent(inout), optional :: sites_outside ! were there sites outside mesh?
    real(8), allocatable :: cnt_(:,:)

    integer :: i        ! loop index for local fission sites
    integer :: n_sites  ! size of bank array
    integer :: n        ! number of energy groups / size
    integer :: mesh_bin ! mesh bin
    integer :: e_bin    ! energy bin
#ifdef OPENMC_MPI
    integer :: mpi_err  ! MPI error code
#endif
    logical :: outside  ! was any site outside mesh?

    ! initialize variables
    allocate(cnt_(size(cnt,1), size(cnt,2)))
    cnt_ = ZERO
    outside = .false.

    ! Set size of bank
    if (present(size_bank)) then
      n_sites = int(size_bank,4)
    else
      n_sites = size(bank_array)
    end if

    ! Determine number of energies in group structure
    if (present(energies)) then
      n = size(energies) - 1
    else
      n = 1
    end if

    ! loop over fission sites and count how many are in each mesh box
    FISSION_SITES: do i = 1, n_sites
      ! determine scoring bin for entropy mesh
      call m % get_bin(bank_array(i) % xyz, mesh_bin)

      ! if outside mesh, skip particle
      if (mesh_bin == NO_BIN_FOUND) then
        outside = .true.
        cycle
      end if

      ! determine energy bin
      if (present(energies)) then
        if (bank_array(i) % E < energies(1)) then
          e_bin = 1
        elseif (bank_array(i) % E > energies(n + 1)) then
          e_bin = n
        else
          e_bin = binary_search(energies, n + 1, bank_array(i) % E)
        end if
      else
        e_bin = 1
      end if

      ! add to appropriate mesh box
      cnt_(e_bin, mesh_bin) = cnt_(e_bin, mesh_bin) + bank_array(i) % wgt
    end do FISSION_SITES

#ifdef OPENMC_MPI
    ! collect values from all processors
    n = size(cnt_)
    call MPI_REDUCE(cnt_, cnt, n, MPI_REAL8, MPI_SUM, 0, mpi_intracomm, mpi_err)

    ! Check if there were sites outside the mesh for any processor
    if (present(sites_outside)) then
      call MPI_REDUCE(outside, sites_outside, 1, MPI_LOGICAL, MPI_LOR, 0, &
           mpi_intracomm, mpi_err)
    end if
#else
    sites_outside = outside
    cnt = cnt_
#endif

  end subroutine count_bank_sites

end module mesh
