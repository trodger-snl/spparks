# potts/rx/kk: static recrystallization of a deformed 3d grain structure
#
# Deformed grains (spins 1..1000) grow at H = 0 first; then every site gets
# stored energy H = 4 and nuclei appear continuously (site-saturated decks
# set i1 > nspins, d1 0 on the nucleus sites instead). Xrx in the stats is
# the recrystallized fraction.

seed		 56789

app_style	 potts/rx/kk 1000 nucleation 0.0002 0.5

lattice		 sc/26n 1.0
region		 box block 0 32 0 32 0 32
create_box	 box
create_sites	 box
set		 i1 range 1 1000
set		 d1 value 0.0

sweep		 color/strict
temperature	 0.5

diag_style	 energy
stats		 10.0
run		 20.0

set		 d1 value 4.0
dump		 1 text 100.0 dump.potts_rx id i1 d1
run		 100.0
