# SPPARKS Potts Quaternion Kokkos test
# Test AppPottsQuaternionKokkos with cubic symmetry

variable Q equal 100

seed		 56789

# Use Kokkos version
app_style	 potts/quaternion/kk ${Q} cubic

dimension	 3
lattice		 sc/26n 1.0
region		 box block 0 30 0 30 0 30
create_box	 box
create_sites	 box
set		 site range 1 ${Q}

sweep		 random
sector		 yes

diag_style       energy

temperature	 0.25

stats            1.0
dump             1 text 10.0 dump.potts_quaternion.kk

run              30.0
