import os
import json
import argparse
parser = argparse.ArgumentParser(description='Compare sample list with actually used samples in VHcc and create used list of samples. \
                                              Note: You should only fetch files via DAS afterwards. Do both before running the processor on unnecessary files.')
parser.add_argument('-n', '--name', default=r'mcsamples_2017_higgs', help='Name of the file containing samples (default: %(default)s)')
args = parser.parse_args()

fset = []

for y in ['2016','2017','2018']:
    if y in args.name:
        year = y
samples = '../metadata/sample_info_' + year

def read_VHcc_dir(files):
    f = open(files)
    data = json.load(f)
    return data.keys()

VHcc_dirs = read_VHcc_dir(samples+'.json')

with open('../samples/'+args.name+'.txt') as fp: 
    lines = fp.readlines() 
    for line in lines:
        if line[0] == '\n':
            fset.append(line)
        elif (line.split('/')[1]).split('/')[0] in VHcc_dirs:
            fset.append(line)
        else:
            fset.append('#'+line)

with open('../samples/'+args.name+'_used.txt', 'w') as f:
    for line in fset:
        f.write(line)
        #f.write('\n')
