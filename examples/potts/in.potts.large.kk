# Potts Kokkos smoke test on a 60^3 box (gfx1151 / HIP)
# Run:  build-verify/spparks -in examples/potts/in.potts.large.kk   (any KOKKOS build)
# Confirm GPU selection with: spparks --kokkos-print-configuration -in ...
seed 12345
app_style potts/kk 100

dimension 3
lattice sc/26n 1.0
region box block 0 60 0 60 0 60
create_box box
create_sites box
set site range 1 100

# KOKKOS apps require color sweeping (not 'sweep random') for correct GPU execution
sweep color
sector yes

temperature 0.5

diag_style energy
stats 1.0

run 5.0
