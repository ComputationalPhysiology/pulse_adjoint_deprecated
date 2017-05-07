import sys, yaml

inputfile = sys.argv[1]

d = yaml.load(open(inputfile, "rb"))

if sys.argv[2] == "outdir":
    sys.stdout.write(d["outdir"])

if sys.argv[2] == "mesh":
    sys.stdout.write(d["Patient_parameters"]["mesh_path"])
if sys.argv[2] == "pressure":
    sys.stdout.write(d["Patient_parameters"]["pressure_path"])
if sys.argv[2] == "echo":
    sys.stdout.write(d["Patient_parameters"]["echo_path"])
