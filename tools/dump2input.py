# Pizza.py script to convert a SPPARKS dump file to a SPPARKS input file
# assumes dump file is in LAMMPS dump format
# input file is used by lattice apps with "input" keyword in app_style

# Syntax: pizza.py -f dump2input.py dumpfile N inputfile
# N = timestamp of snapshot to be converted

if len(argv) != 4:
  raise Exception("Syntax: pizza.py -f dump2input.py dumpfile N inputfile")

dumpfile = argv[1]
ntime = int(argv[2])
inputfile = argv[3]

# next sections extract dump snapshot and write SPPARKS file
# edit number of columns and column assignments if needed

# example for on-lattice file with just ID and lattice value

d = dump(dumpfile)
d.map(1,"id",2,"lattice")
id,lattice = d.vecs(ntime,"id","lattice")

fp = open(inputfile,"w")
print("SPPARKS input file from dump file, time =",ntime,file=fp)
print(len(id),1,file=fp)
print(file=fp)
for i in range(len(id)):
  print(int(id[i]),int(lattice[i]),file=fp)
fp.close()

# example for off-lattice file with ID, x, y, z, site value

#d = dump(dumpfile)
#d.map(1,"id",2,"site",3,"x",4,"y",5,"z")
#id,x,y,z,site = d.vecs(ntime,"id","x","y","z","site")

#fp = open(inputfile,"w")
#print("SPPARKS input file from dump file, time =",ntime,file=fp)
#print(len(id),1,file=fp)
#print(file=fp)
#for i in range(len(id)):
#  print(int(id[i]),int(site[i]),float(x[i]),float(y[i]),float(z[i]),file=fp)
#fp.close()
