# Use like:
# python fetch.py -i ../samples/mcsamples_2017_higgs_used.txt -s phys03 -o mcsamples_2017_higgs_used
import os
import json
import argparse
parser = argparse.ArgumentParser(description='Create json with individual paths for specified datasets.\
                                              Note: You should compare the available datasets with the necessary ones first to avoid running on unused samples.')
parser.add_argument('-i', '--input', default=r'singlemuon', help='List of samples in DAS (default: %(default)s)')
parser.add_argument('-s', '--site', default=r'global', help='Site (default: %(default)s)')
parser.add_argument('-o', '--output', default=r'singlemuon', help='Site (default: %(default)s)')
args = parser.parse_args()
fset = []

with open(args.input) as fp: 
    lines = fp.readlines() 
    for line in lines:
        if line[0] != '\n' and line[0] != '#':
            fset.append(line)

fdict = {}

instance = 'prod/'+args.site


xrd = 'root://xrootd-cms.infn.it//'

for dataset in fset:
    print(dataset)
    flist = os.popen(("/cvmfs/cms.cern.ch/common/dasgoclient -query='instance={} file dataset={}'").format(instance,fset[fset.index(dataset)].rstrip())).read().split('\n')
    dictname = dataset.rstrip()
    if dictname not in fdict:
        fdict[dictname] = [xrd+f for f in flist if len(f) > 1]
    else: #needed to collect all data samples into one common key "Data" (using append() would introduce a new element for the key)
        fdict[dictname].extend([xrd+f for f in flist if len(f) > 1])

#pprint.pprint(fdict, depth=1)

with open('../metadata/%s.json'%(args.output), 'w') as fp:
    json.dump(fdict, fp, indent=4)
