# potts/anneal/kk: recrystallization anneal of a deformed cubic polycrystal
#
# Stands in for an additive/texture state (load one with set ... stitch:
# i1 i2 d1..d4 d8 d9). Here grains grow first at H = 0 (to ~6 sites
# across: in much finer grains, boundary fluctuations at T = 2 mix H
# between grains before nuclei grow), each with an
# orientation keyed to its label; the solidification G (d8, K/m) and V
# (d9, m/s) are set by hand, finer cells (higher G) in the top half.
# stored_energy maps PDAS -> dislocation density -> H, scaled so that
# rho_ref = 2e14 m^-2 gives H = 6 (the linear window on sc/26n at T = 2):
# H ~ 3 below, ~5.7 in the top half.

seed		 56789

app_style	 potts/anneal/kk 100000 cubic 15.0 &
		 recovery 0.002 nucleation 0.000005 2.0 boundary radius 4

lattice		 sc/26n 1.0
region		 box block 0 32 0 32 0 32
create_box	 box
create_sites	 box
set		 i1 range 1 3000
set		 i2 value 3
set		 d5 value 0.0

sweep		 color/strict
temperature	 2.0
stats		 10.0
run		 30.0

region		 top block INF INF INF INF 16 INF
set		 d8 value 1.0e6
set		 d8 value 4.0e6 region top
set		 d9 value 0.1
stored_energy	 pdas 428.44 1.0e13 0.0175 2.54e-10 2.0e14 6.0

diag_style	 array d5 mean
dump		 1 text 100.0 dump.potts_anneal id i1 d1 d2 d3 d4 d5
run		 100.0
