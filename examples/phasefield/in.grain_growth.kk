# Fan-Chen grain growth, 8 order parameters from small random noise:
# grains nucleate from the noise and coarsen; the energy falls

seed             56789

#                          dt   model        N L kappa alpha beta gamma
app_style        phasefield/kk 0.02 grain_growth 8 1 1     1     1    1

dimension        2
lattice          sq/4n 1.0
region           box block 0 128 0 128 -0.5 0.5
create_box       box
create_sites     box

set              d1 range 0 0.1
set              d2 range 0 0.1
set              d3 range 0 0.1
set              d4 range 0 0.1
set              d5 range 0 0.1
set              d6 range 0 0.1
set              d7 range 0 0.1
set              d8 range 0 0.1

diag_style       energy
stats            20.0
dump             1 text 100.0 dump.grain_growth id x y d1 d2 d3 d4 d5 d6 d7 d8
run              200.0
