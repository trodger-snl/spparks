# potts/rx/kk: dynamic recrystallization of a 3d grain structure
#
# Every grain hardens, dH/dt = 0.1 - 0.02 H (saturating at H = 5), while
# sites on grain boundaries with H >= 3 nucleate new grains, recrystallized
# grains included: the grain size reaches a steady state while new
# generations of grains replace the old. mean(d1) is the mean stored
# energy; Xrx the fraction ever recrystallized.

seed		 56789

app_style	 potts/rx/kk 1000 hardening 0.1 0.02 &
		 nucleation 0.0005 3.0 boundary

lattice		 sc/26n 1.0
region		 box block 0 32 0 32 0 32
create_box	 box
create_sites	 box
set		 i1 range 1 1000
set		 d1 value 0.0

sweep		 color/strict
temperature	 0.5

diag_style	 energy
diag_style	 array d1 mean
stats		 20.0
dump		 1 text 200.0 dump.potts_drx id i1 d1
run		 200.0
