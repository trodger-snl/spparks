# SPPARKS Potts/Neigh tests on generic lattice - Kokkos version
# Uses neighbor-based rejection with null bin

seed		 56789

# Use Kokkos-enabled Potts Neigh app
app_style	 potts/neigh/kk 100

dimension	 3
lattice		 sc/26n 1.0
region		 box block 0 20 0 20 0 20
create_box	 box
create_sites	 box
set		 site range 1 100

sweep		 color
sector		 yes

diag_style       energy

temperature	 0.0

stats            10.0
dump             1 text 10.0 dump.potts.neigh.kk
diag_style       cluster delt 1.0 stats no logfreq 9 10.0 filename cluster.neigh.kk.dat

run              100.0
