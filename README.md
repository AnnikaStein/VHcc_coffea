# VHcc_coffea
VHcc analysis framework, using coffea similar to https://github.com/Ming-Yan/Hpluscharm and https://github.com/cms-btv-pog/BTVNanoCommissioning

Based on (restricted access!): https://github.com/mastrolorenzo/AnalysisTools-1 and https://gitlab.cern.ch/aachen-3a/vhcc-nano

## Setup
! This environment should run under `bash` shell ! 
### Coffea installation with Miniconda
#### only first time
```
git clone git@github.com:AnnikaStein/VHcc_coffea.git
```
For installing Miniconda, see also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Local-or-remote
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run and follow instructions on screen
bash Miniconda3-latest-Linux-x86_64.sh
```
NOTE: always make sure that conda, python, and pip point to local Miniconda installation (`which conda` etc.).

You can either use the default environment `base` or create a new one:
```
# create new environment with python 3.7, e.g. environment of name `coffea`
conda create --name coffea python3.7
# activate environment `coffea`
conda activate coffea
```
Install coffea, xrootd, and more:
```
pip install git+https://github.com/CoffeaTeam/coffea.git #latest published release with `pip install coffea`
conda install -c conda-forge xrootd
conda install -c conda-forge ca-certificates
conda install -c conda-forge ca-policy-lcg
conda install -c conda-forge voms
conda install -c conda-forge dask-jobqueue
conda install -c anaconda bokeh 
conda install -c conda-forge 'fsspec>=0.3.3'
conda install dask
conda install -c conda-forge parsl
```
To avoid some import error:
```
conda install -c conda-forge vector
```
Setup things to use x509 proxy
```bash
# There are some dependencies which are currently not available as packages, so we'll have to copy them from `/cvmfs/grid.cern.ch/etc/grid-security/`
   
    mkdir ~/.grid-security
    scp -r lxplus:/cvmfs/grid.cern.ch/etc/grid-security/* ~/.grid-security
   
    
#All that's left is to point `voms-proxy-init` to the right vomses directory
   
    voms-proxy-init --voms cms --vomses ~/.grid-security/vomses
```


### Other installation options for coffea
See https://coffeateam.github.io/coffea/installation.html

For example, I find jupyterlab convenient:
```
conda install jupyter
conda install -c conda-forge jupyterlab
```

And I used this one as well:
```
pip install coffea[dask,spark,parsl]
```
### Running jupyter remotely
See also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Remote-jupyter
1. On your local machine, edit `.ssh/config`:
```
Host lxplus*
  HostName lxplus7.cern.ch
  User <your-user-name>
  ForwardX11 yes
  ForwardAgent yes
  ForwardX11Trusted yes
Host *_f
  LocalForward localhost:8800 localhost:8800
  ExitOnForwardFailure yes
```
2. Connect to remote with `ssh lxplus_f`
3. Start a jupyter notebook:
```
jupyter notebook --ip=127.0.0.1 --port 8800 --no-browser
```




## Summary: what to do each time when running 
1. activate coffea
`conda activate coffea`
2. setup the proxy
`voms-proxy-init --voms cms --vomses ~/.grid-security/vomses --valid 192:00`
3. Run the framework, see below



## Structure

Each workflow can be a separate "processor" file, creating the mapping from NanoAOD to
the histograms we need. Workflow processors can be passed to the `runner.py` script 
along with the fileset these should run over. Multiple executors can be chosen 
(for now iterative - one by one, uproot/futures - multiprocessing and dask-slurm - TBC). 

For now only the selections and histogram producer are available.

To test a small set of files to see whether the workflows run smoothly, run:
```
python runner.py  --json metadata/mcsamples_higgs_2017.json --limit 1 --wf Zll
```
Or with one specific sample only:
```
python runner.py  --json metadata/mcsamples_higgs_2017.json --limit 1 --wf Zll --only /ZH_HToCC_ZToLL_M125_13TeV_powheg_pythia8/coli-NanoTuples-30Apr2020_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_v14-v2-6f7c69ffdbb83072d4913e5f3cf0008f/USER
```
Similar with data, example:
```
python runner.py  --json metadata/datasamples_2017.json --limit 1 --wf Zll --only /DoubleMuon/hqu-NanoTuples-30Apr2020_Run2017F-31Mar2018-v1-b2e5aecd7d318124ef1b7f48a4318be4/USER
```
- Samples are in `metadata`
- Corrections in `utils/correction.py` 

### Scale out @lxplus (not recommended)

```
python runner.py  --json metadata/mcsamples_higgs_2017.json --limit 1 --wf Zll -j ${njobs} --executor dask/lxplus
``` 
### Scale out @DESY/lxcluster Aachen

```
python runner.py  --json metadata/mcsamples_higgs_2017.json --limit 1 --wf Zll -j ${njobs} --executor parsl/condor --memory ${memory_per_jobs}
```
### Scale out @HPC Aachen

```
python runner.py  --json metadata/mcsamples_higgs_2017.json --limit 1 --wf Zll -j ${njobs} --executor parsl/slurm
```



