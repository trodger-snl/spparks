# SPPARKS Potts test - Larger system for benchmarking Kokkos

seed		 56789

# Use Kokkos-enabled Potts app
app_style	 potts/kk 1000

dimension	 3
lattice		 sc/26n 1.0
region		 box block 0 50 0 50 0 50
create_box	 box
create_sites	 box
set		 site range 1 1000

sweep		 raster
sector		 no   # Kokkos apps don't support sectors yet

diag_style       energy

temperature	 0.0

stats            10.0
run              50.0
