# SPPARKS Potts tests on generic lattice - Kokkos version

seed		 56789

# Use Kokkos-enabled Potts app
app_style	 potts/kk 100

dimension	 3
lattice		 sc/26n 1.0
region		 box block 0 20 0 20 0 20
create_box	 box
create_sites	 box
set		 site range 1 100

sweep		 color
sector		 yes

#solve_style      tree
#solve_style      group 26 1
#sector		 yes

diag_style       energy

temperature	 0.0

stats            10.0
dump             1 text 10.0 dump.potts.kk

diag_style       cluster delt 1.0 stats no logfreq 9 10.0 filename cluster.kk.dat
dump		 2 image 1.0 dump.potts-kk.*.jpg site site crange 1 100 drange 1 1 shape cube box no 1

run              100.0
