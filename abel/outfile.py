import sys, yaml

inputfile = sys.argv[1]

d = yaml.load(open(inputfile, "rb"))

sys.stdout.write(d["outdir"])
