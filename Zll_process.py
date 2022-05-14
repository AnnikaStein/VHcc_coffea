import csv
from curses import meta
from dataclasses import dataclass
import gzip
import pickle, os, sys, mplhep as hep, numpy as np
from select import select

#from matplotlib.pyplot import jet

#import coffea
from coffea import hist, processor
from coffea.nanoevents.methods import vector
import awkward as ak
from utils.correction import jec,muSFs,eleSFs,init_corr
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights, PackedSelection
from functools import partial
# import numba
from helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4

def mT(obj1,obj2):
    return np.sqrt(2.*obj1.pt*obj2.pt*(1.-np.cos(obj1.phi-obj2.phi)))
def flatten(ar): # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)
def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, year="2017"):    
        self._year=year
        
        # paths from table 1 and 2 of the AN_2020_235
        
        # l l
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L3328
        self._mumu_hlt = {
            '2016': [
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ'
            ],
            '2017': [
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',#allowMissingBranch=1
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'#allowMissingBranch=1
            ],
            '2018': [
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',#allowMissingBranch=1 but this is the only used one in 2018?!
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'#allowMissingBranch=1
            ],
        }   
    
        self._ee_hlt = {
            '2016': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'
            ],
            '2017': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                #'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ' # not in VHccAnalysis code
            ],
            '2018': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'
            ],
        }  
        
        '''
        # l nu
        self._munu_hlt = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24'
            ],
            '2017': [
                'IsoMu24',
                'IsoMu27'
            ],
            '2018': [
                'IsoMu24',
                'IsoMu27'
            ],
        }   
    
        self._enu_hlt = {
            '2016': [
                'Ele27_eta2p1_WPTight_Gsf'
            ],
            '2017': [
                'Ele32_WPTight_Gsf_L1DoubleEG',
                'Ele32_WPTight_Gsf'
            ],
            '2018': [
                'Ele32_WPTight_Gsf_L1DoubleEG',
                'Ele32_WPTight_Gsf'#allowMissingBranch=1
            ],
        }  
        
        # nu nu
        self._nunu_hlt = {
            '2016': [
                'PFMET110_PFMHT110_IDTight',
                #'PFMET110_PFMHT120_IDTight', # found in hltbranches_2016.txt but not in AN, maybe redundant?
                'PFMET170_NoiseCleaned',#allowMissingBranch=1
                'PFMET170_BeamHaloCleaned',#allowMissingBranch=1
                'PFMET170_HBHECleaned'
            ],
            '2017': [
                'PFMET110_PFMHT110_IDTight',
                'PFMET120_PFMHT120_IDTight',
                'PFMET120_PFMHT120_IDTight_PFHT60',#allowMissingBranch=1
                'PFMETTypeOne120_PFMHT120_IDTight'
            ],
            '2018': [
                'PFMET110_PFMHT110_IDTight',
                'PFMET120_PFMHT120_IDTight',
                'PFMET120_PFMHT120_IDTight_PFHT60'#allowMissingBranch=1
            ],
        } 
        
        '''
        
        # differences between UL and EOY
        # see https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        # also look at sec. 3.7.2
        self._met_filters = {
            '2016': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    'eeBadScFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'eeBadScFilter', # not suggested in EOY MC
                ],
            },
            '2017': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    #'eeBadScFilter', # not suggested in EOY MC
                    'ecalBadCalibFilterV2',
                ],
            },
            '2018': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    #'eeBadScFilter', # not suggested in EOY MC
                    'ecalBadCalibFilterV2',
                ],
            },
        }
        
        
        # Most likely there are others for EOY, given that these ones contain UL or Legacy in the name...
        self._lumiMasks = {
            '2016': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
            '2017': LumiMask('data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
            '2018': LumiMask('data/Lumimask/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
        }
        
        self._corr = init_corr(year)
        
        # Axes: Cat - what it is, a type of something, described with words
        #       Bin - how much of something, numerical things
        #
        #   --> What follows here are "general" axes that are not a priori connected to objects,
        #       but serve as building blocks to be used multiple times
        #   --> Explains my old "DeltaR between what?" comment, namely because explicit usage only comes later
        
        
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        
        # ToDo: look up what this means, what the numbers represent.
        #       Also, is it necessary for VHcc? Is it for the c jet?
        #       Is it just the weird binning that confuses me and
        #       maybe just the hadron or parton flavor in BTV nomenclature? 0, 4, 5
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        
        lepflav_axis = hist.Cat("lepflav",['ee','mumu'])
        
        '''
        # Outdated idea: probably useful to define high and low Vpt regions?
        Zll_vpt_region_axis = hist.Cat("Zll_vpt_region",['low_vpt','high_vpt'])
        '''
        # WIP: could add the split in vpt here instead because I assume these are all disjoint by construction
        # WIP: region is used later again, could probably define here already and then just refer to it
        #regions = ['SR','SR_Zcc','top_antitop','Z_plus_c','Z_plus_b','Z_plus_l','W_plus_c','W_plus_l']
        # not yet sure how to handle the Zcc one
        regions = ['SR_2LL','SR_2LH','Zcc','top_antitop','Z_plus_c','Z_plus_b','Z_plus_l','W_plus_c','W_plus_l']
        region_axis = hist.Cat("region",regions)
        
        # these can stay how they are for the moment, just make sure sufficient information is stored later
        # Events
        njet_axis  = hist.Bin("nj",  r"N jets",      [0,1,2,3,4,5])
        nbjet_axis = hist.Bin("nbj", r"N b jets",    [0,1,2,3,4,5])            
        ncjet_axis = hist.Bin("ncj", r"N c jets",    [0,1,2,3,4,5])
        # kinematic variables       
        pt_axis   = hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 300)
        eta_axis  = hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 50, 0, 300)
        mt_axis =  hist.Bin("mt", r" $m_{T}$ [GeV]", 30, 0, 300)
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
        # MET vars
        signi_axis = hist.Bin("significance", r"MET $\sigma$",20,0,10)
        covXX_axis = hist.Bin("covXX",r"MET covXX",20,0,10)
        covXY_axis = hist.Bin("covXY",r"MET covXY",20,0,10)
        covYY_axis = hist.Bin("covYY",r"MET covYY",20,0,10)
        sumEt_axis = hist.Bin("sumEt", r" MET sumEt", 50, 0, 300)
        
        # axis.StrCategory([], name='region', growth=True),
        #disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        # As far as I can tell, we only need DeepFlav currently
        #disc_list = ['btagDeepFlavCvB','btagDeepFlavCvL']
        # BUT: CvB and CvL not available, but can be recalculated
        disc_list = ['btagDeepFlavC','btagDeepFlavB']
        btag_axes = []
        for d in disc_list:
            # ToDo: find out why -1 bin is irrelevant here
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
            
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis, region_axis, njet_axis),
                'nbj' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, nbjet_axis),
                'ncj' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, ncjet_axis),
                'hj_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, dr_axis),
                'MET_sumEt' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, sumEt_axis),
                'MET_significance' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, signi_axis),
                'MET_covXX' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, covXX_axis),
                'MET_covXY' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, covXY_axis),
                'MET_covYY' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, covYY_axis),
                'MET_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, phi_axis),
                'MET_pt' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, pt_axis),
                'mT1' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, mt_axis),
                'mT2' : hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, mt_axis),
                'mTh':hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, mt_axis),
                'dphi_lep1':hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, phi_axis),
                'dphi_lep2':hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, phi_axis),
                'dphi_ll':hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, phi_axis),
                # already put these lines here to refer to in future (with Zll_vpt_region_axis)
                #     although it is possible to just split SR inside the region axis in two
                #     guess that was an initial misunderstanding by me --> just ignore
                #'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis, Zll_vpt_region_axis, region_axis, njet_axis),
                #'nbj' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, nbjet_axis),
                #'ncj' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, ncjet_axis),
                #'hj_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, dr_axis),
                #'MET_sumEt' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, sumEt_axis),
                #'MET_significance' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, signi_axis),
                #'MET_covXX' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, covXX_axis),
                #'MET_covXY' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, covXY_axis),
                #'MET_covYY' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, covYY_axis),
                #'MET_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, phi_axis),
                #'MET_pt' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, pt_axis),
                #'mT1' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, mt_axis),
                #'mT2' : hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, mt_axis),
                #'mTh':hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, mt_axis),
                #'dphi_lep1':hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, phi_axis),
                #'dphi_lep2':hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, phi_axis),
                #'dphi_ll':hist.Hist("Counts", dataset_axis, lepflav_axis, Zll_vpt_region_axis, region_axis, phi_axis),
            }
        # ToDo: find out what jetpn stands for, and why it is only referenced in the loop below, but afterwards only commented-out
        #       is it ordering jets by ParticleNet, DeepFlavour etc.? Saw also pt and csv in another workflow.
        objects=['jetflav','jetpn','lep1','lep2','ll']
        
        for i in objects:
            if 'jet' in i: 
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, flav_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, flav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, flav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, flav_axis, mass_axis)
            else:
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis, mass_axis)
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["jetflav_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis, region_axis,flav_axis, axis)
            
        self.event_hists = list(_hist_event_dict.keys())
    
        # Seems like here we define what will be stored and how it will be stored in the final output:
        # - explains why sometimes, there is only a numerical value, and in other cases a histogram instead
        self._accumulator = processor.dict_accumulator(
            {**_hist_event_dict,   
             'cutflow': processor.defaultdict_accumulator(
                 # we don't use a lambda function to avoid pickle issues
                 partial(processor.defaultdict_accumulator, int))
            })
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        # Q: could there be MC that does not have this attribute? Or is it always the case?
        isRealData = not hasattr(events, "genWeight")
        
        # length of events is used so many times later on, probably useful to just save it here and then refer to that
        nEvents = len(events)
        print('Number of events: ', nEvents)
        
        # As far as I understand, this looks like a neat way to give selections a name,
        # while internally, there are boolean arrays for all events
        selection = PackedSelection()
        
        
        # this is either counting events in data with weight 1, or weighted (MC)
        if isRealData:
            output['sumw'][dataset] += nEvents
        else:
            # instead of taking the weights themselves, the sign is used:
            # https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/7
            # although I initially had the same concerns as those raised in the thread,
            # if not only the sign is different, but also the absolute values between events
            # somehow it seems to average out, although I don't see why this is guaranteed
            # must have to do with "LO without interference" where the values are indeed same
            # technically it should not be necessary for EOY samples? And what if there are
            # samples where these genWeights do vary in scale, not only sign?
            output['sumw'][dataset] += ak.sum(events.genWeight/abs(events.genWeight))
            
            
        req_lumi=np.ones(nEvents, dtype='bool')
        if isRealData: 
            req_lumi=self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumi',ak.to_numpy(req_lumi))
        del req_lumi
        
        
        # AS: sort of the same thing as above, but now per entry
        weights = Weights(nEvents, storeIndividual=True)
        if isRealData:
            weights.add('genweight',np.ones(nEvents))
        else:
            weights.add('genweight',events.genWeight/abs(events.genWeight))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
            
            
        ##############
        if isRealData:
            output['cutflow'][dataset]['all']  += nEvents
        else:
            output['cutflow'][dataset]['all']  += ak.sum(events.genWeight/abs(events.genWeight))
            
        
        #trigger_met = np.zeros(nEvents, dtype='bool')

        trigger_ee = np.zeros(nEvents, dtype='bool')
        trigger_mm = np.zeros(nEvents, dtype='bool')

        #trigger_e = np.zeros(nEvents, dtype='bool')
        #trigger_m = np.zeros(nEvents, dtype='bool')
        
        #for t in self._nunu_hlt[self._year]:
        #    # so that already seems to be the check for whether the path exists in the file or not
        #    if t in events.HLT.fields:
        #        trigger_met = trigger_met | events.HLT[t]

        for t in self._mumu_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]

        for t in self._ee_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]

        #for t in self._munu_hlt[self._year]:
        #    if t in events.HLT.fields:
        #        trigger_m = trigger_m | events.HLT[t]

        #for t in self._emu_hlt[self._year]:
        #    if t in events.HLT.fields:
        #        trigger_e = trigger_e | events.HLT[t]
        
        
        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mm))
        
        
        # apart from the comments above about EOY/UL, should be fine
        metfilter = np.ones(nEvents, dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter
        
        
        
        
        
        # =================================================================================
        #
        # #                       Reconstruct and preselect leptons
        #
        # ---------------------------------------------------------------------------------
        
        
        # Adopt from https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L3369-L3440
        # https://gitlab.cern.ch/aachen-3a/vhcc-nano/-/blob/master/VHccProducer.py#L345-389
        
        # ## Muon cuts
        ## muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        #event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1, ascending=False)]
        event_mu = events.Muon
        # I expect that there is a bunch of corrections that are not yet implemented here, following the discussions on Mon & Wed
        # looseId >= 1 or looseId seems to be the same...
        musel = ((event_mu.pt > 20) & (abs(event_mu.eta) < 2.4) & (event_mu.looseId >= 1) & (event_mu.pfRelIso04_all<0.25))
        # but 25GeV and 0.06 for 1L, xy 0.05 z 0.2, &(abs(event_mu.dxy)<0.06)&(abs(event_mu.dz)<0.2) and tightId for 1L
        event_mu = event_mu[musel]
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1, ascending=False)]
        # Q: is 13 used here because of some PDG code?
        event_mu["lep_flav"] = 13*event_mu.charge
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        
        # ## Electron cuts
        ## # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        #event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e = events.Electron
        elesel = ((event_e.pt > 20) & (abs(event_e.eta) < 2.5) & (event_e.mvaFall17V2Iso_WP90==1) & (event_e.pfRelIso03_all<0.25))
        # but 30GeV and WP80 for 1L
        event_e = event_e[elesel]
        event_e = event_e[ak.argsort(event_e.pt, axis=1,ascending=False)]
        # Q: see above
        event_e["lep_flav"] = 11*event_e.charge
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)
        # sorting after selecting should be faster (less computations on average, no?)
   
        
        # for this channel (Zll / 2L)
        selection.add('lepsel',ak.to_numpy((nele==2)|(nmu==2)))
        
        
        
        #### build lepton pair(s) (I guess to reconstruct Z)
        good_leptons = ak.with_name(
                ak.concatenate([ event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate", )
        good_leptons = good_leptons[ak.argsort(good_leptons.pt, axis=1,ascending=False)]
        leppair = ak.combinations(
                good_leptons,
                n=2,
                replacement=False,
                axis=-1,
                fields=["lep1", "lep2"],
            )
        # print(leppair.tolist())
        #print(leppair.type)
        ll_cand = ak.zip({
                    "lep1" : leppair.lep1,
                    "lep2" : leppair.lep2,
                    "pt": (leppair.lep1+leppair.lep2).pt,
                    "eta": (leppair.lep1+leppair.lep2).eta,
                    "phi": (leppair.lep1+leppair.lep2).phi,
                    "mass": (leppair.lep1+leppair.lep2).mass,
                    }, with_name="PtEtaPhiMLorentzVector"
                )
        ll_cand = ak.pad_none(ll_cand,1,axis=1)
        
        if (ak.count(ll_cand.pt)>0):
            ll_cand  = ll_cand[ak.argsort(ll_cand.pt, axis=1,ascending=False)]
            
            
            
            
        # =================================================================================
        #
        # #                       Reconstruct and preselect jets
        #
        # ---------------------------------------------------------------------------------
        
        # Apply correction:
        jets =  jec(events,events.Jet,dataset,self._year,self._corr)
        
        
        # This was necessary for the FSR code
        #jets = jets.mask[ak.num(jets) > 2]
        
        
        
        # For EOY: recalculate CvL & CvB here, because the branch does not exist in older files
        # adapted from PostProcessor
        def deepflavcvsltag(jet):
            btagDeepFlavL = 1.-(jet.btagDeepFlavC+jet.btagDeepFlavB)
            return ak.where((jet.btagDeepFlavB >= 0.) & (jet.btagDeepFlavB < 1.) & (jet.btagDeepFlavC >= 0.) & (btagDeepFlavL >= 0.),
                            jet.btagDeepFlavC/(1.-jet.btagDeepFlavB),
                            (-1.) * ak.ones_like(jet.btagDeepFlavB))
        
        def deepflavcvsbtag(jet):
            btagDeepFlavL = 1.-(jet.btagDeepFlavC+jet.btagDeepFlavB)
            return ak.where((jet.btagDeepFlavB > 0.) & (jet.btagDeepFlavC > 0.) & (btagDeepFlavL >= 0.),
                            jet.btagDeepFlavC/(jet.btagDeepFlavC+jet.btagDeepFlavB),
                            (-1.) * ak.ones_like(jet.btagDeepFlavB))
        
        # Alternative ways:
        # - depending on the Nano version, there might already be bTagDeepFlavCvL available
        # - one could instead use DeepCSV via bTagDeepCvL
        # - not necessarily use CvL, other combination possible ( CvB | pt | BDT? )
        
        jets = jets[ak.argsort(deepflavcvsltag(jets), axis=1, ascending=False)]

        
        # Jets are considered only if the following identification conditions hold, as mentioned in AN
        # - Here is some documentation related to puId and jetId:
        #     https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
        #     https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID
        jet_conditions = ((abs(jets.eta) < 2.4) & (jets.pt > 20) & (jets.puId > 0)) \
                     | ((jets.pt>50) & (jets.jetId>5))
        # Count how many jets exist that pass this selection
        njet = ak.sum(jet_conditions,axis=1)
        selection.add('jetsel',ak.to_numpy(njet>=2))
        
        # FSR jets are selected with slightly different criteria
        fsr_conditions = (abs(jets.eta) < 3) & (jets.pt > 20) \
                        & ak.all(jets.metric_table(ll_cand.lep1)>0.2) & ak.all(jets.metric_table(ll_cand.lep2)>0.2)
        # Take the first two jets that pass the criteria and check the remaining ones,
        # as well as potentially others, to get FSR jets:
        pick2 = jets[ak.pad_none(ak.local_index(jets, 1)[jet_conditions], 2)[:, :2]]
        others = jets[ak.concatenate([ak.pad_none(ak.local_index(jets, 1)[jet_conditions], 2)[:, 2:], 
                                    ak.local_index(jets, 1)[(~jet_conditions) & (fsr_conditions)]
                                   ], axis=1)]
        
        
        #print(pick2.type)
        #print(rest.type)
        #print(others.type)
        
        # This first version of the FSR recovery is not able to find the remaining FSR candidates that do not fulfill the
        # jet_conditions and therefore not considered further.
        #_limit = 3
        #_consider_only = 2
        #leading, subleading, others = jets[:_limit, 0], jets[:_limit, 1], jets[:_limit, 2:2+_consider_only]

        def find_fsr(leading, subleading, others, threshold=3):
            mval1, (a1, b) = leading.metric_table(others, return_combinations=True)
            mval2, (a2, b) = subleading.metric_table(others, return_combinations=True)

            def res(mval, out):
                order = ak.argsort(mval, axis=-1)
                #print(order.type)
                #print(out.type)
                #print(mval.type)
                return out[order], mval[order]

            out1, metric1 =  res(mval1, b)
            out2, metric2 =  res(mval2, b)

            out1 = out1.mask[(metric1 <= threshold) & (metric1 < metric2)]
            #out2 = out2.mask[(metric2 <= threshold) & (metric2 < metric1)]
            out2 = out2.mask[(metric1 <= threshold) & (metric2 < metric1)]
            return out1[:, 0, ...], out2[:, 0, ...]

        #leading = pick2[:, 0]
        #subleading = pick2[:, 1]
        
        missing = ~(ak.is_none(pick2[:, 0]) | ak.is_none(pick2[:, 1]))
        pick2 = pick2.mask[missing]
        others = others.mask[missing]

        #print(pick2.type)
        #print(rest.type)
        
        leading, subleading = pick2[:, 0], pick2[:, 1]
        fsr_leading, fsr_subleading = find_fsr(leading, subleading, others, threshold=0.8)

        #print(leading.pt)
        #print((leading + fsr_leading.sum()).pt)
        
        # To explicitly check that adding FSR does indeed have an effect
        #print(ak.sum((leading + fsr_leading.sum()).pt != leading.pt))
        
        #print(leading.type)
        
        # Collect the (sub-)leading jets and their respective FSR jets in a new 4-vector
        leading_with_fsr = ak.zip({
                    "jet1" : leading,
                    "jet2" : fsr_leading.sum(),
                    "pt": (leading + fsr_leading.sum()).pt,
                    "eta": (leading + fsr_leading.sum()).eta,
                    "phi": (leading + fsr_leading.sum()).phi,
                    "mass": (leading + fsr_leading.sum()).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        subleading_with_fsr = ak.zip({
                    "jet1" : subleading,
                    "jet2" : fsr_subleading.sum(),
                    "pt": (subleading + fsr_subleading.sum()).pt,
                    "eta": (subleading + fsr_subleading.sum()).eta,
                    "phi": (subleading + fsr_subleading.sum()).phi,
                    "mass": (subleading + fsr_subleading.sum()).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        

        # =================================================================================
        #
        # #                       Build Higgs candidate w/ or w/o FSR
        #
        # ---------------------------------------------------------------------------------
        
        # Build 4-vector from leading + subleading jets, with or without FSR
        higgs_cand_no_fsr = ak.zip({
                    "jet1" : leading,
                    "jet2" : subleading,
                    "pt": (leading + subleading).pt,
                    "eta": (leading + subleading).eta,
                    "phi": (leading + subleading).phi,
                    "mass": (leading + subleading).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        higgs_cand = ak.zip({
                    "jet1" : leading_with_fsr,
                    "jet2" : subleading_with_fsr,
                    "pt": (leading_with_fsr + subleading_with_fsr).pt,
                    "eta": (leading_with_fsr + subleading_with_fsr).eta,
                    "phi": (leading_with_fsr + subleading_with_fsr).phi,
                    "mass": (leading_with_fsr + subleading_with_fsr).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        
        
        # Not sure if this is even needed in the Zll channel, but it probably won't hurt
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "energy": events.MET.sumEt,
                    }, with_name="PtEtaPhiMLorentzVector"
                )
        
        
        
        
        
        # =================================================================================
        #
        # #                       Actual event selection starts here
        #
        # ---------------------------------------------------------------------------------
        
        
        # Common global requirements in the Zll channel
        # - valid for 2LH and 2LL
        # - valid for any region, no matter if SR or CR
        
        # Q: Not entirely sure about the p_T(H)<250 here, I might be confusing it with 300 GeV, just copied 250 from the AN & AT
        # - If one requires this here, wouldn't the range between 250 and 300 GeV be missing in the combination?
        # - (Assuming boosted is only considering p_T > 300) 
        req_global = ak.any((leppair.lep1.pt>20) & (leppair.lep2.pt>20) \
                        & (ll_cand.mass>75) & (ll_cand.mass<150) \
                        & (ll_cand.pt>50) \
                        & (leading_with_fsr.pt>20) & (subleading_with_fsr.pt>20) \
                        & (higgs_cand.mass<250) \
                        & (leppair.lep1.charge+leppair.lep2.charge==0),  # opposite charge
                        #& (events.MET.pt>20) \
                        #& (make_p4(leppair.lep1).delta_r(make_p4(leppair.lep2))>0.4),
                        axis=-1
            )
        
        
        # Some lines that I copied over, but will most likely not play any role
        #req_sr = ak.any((mT(leppair.lep2,met)>30) & (mT(ll_cand,met)>60)  & (events.MET.sumEt>45),axis=-1) 
        #req_sr = ak.any(
        #
        #    )
        #req_llmass = ak.all((abs(ll_cand.mass-91.18) > 15),axis=-1)
        # print(req_llmass.tolist(),abs(ll_cand.mass-91.18).tolist())
        # print(dataset,abs(ll_cand.mass-91.18).tolist())  
        
        
        
        
        selection.add('global_selection',ak.to_numpy(req_global))
        
        # AS: not sure, but shouldn't it be nele==2 here? Also for H+c
        # also: isn't this pt>13 redundant, given that all e and mu passed the previous cut of pt>13 in H+c, or 20 for VHcc?
        #mask2e =  req_sr&req_global & (nele==1)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)&req_llmass
        
        #mask2e =  req_sr&req_global & (nele==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)&req_llmass
        #mask2mu =  req_sr&req_global & (nmu==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)&req_llmass
        
        mask2e = req_global & (nele==2)
        mask2mu = req_global & (nmu==2)
        
        #mask2lep = [ak.any(tup) for tup in zip(maskemu, mask2mu, mask2e)]
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        
        good_leptons = ak.mask(good_leptons,mask2lep)
       
        
        # output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        
        selection.add('ee',ak.to_numpy(nele==2))
        selection.add('mumu',ak.to_numpy(nmu==2))
        
        
        #req_sr = ak.any((mT(leppair.lep2,met)>30) & (mT(ll_cand,met)>60)  & (events.MET.sumEt>45)&(ak.sum(jet_conditions,axis=-1)>=1),axis=-1) 
        #req_sr1 = ak.any((mT(leppair.lep2,met)>30) & (mT(ll_cand,met)>60) & (abs(ll_cand.mass-91.18)>15) & (events.MET.sumEt>45)&(ll_cand.mass<120),axis=-1) ## due to ttcr1
        #req_sr2 = ak.any((mT(leppair.lep2,met)>30) & (mT(ll_cand,met)>60) & (abs(ll_cand.mass-91.18)>15) & (events.MET.sumEt>45)&(ak.sum(jet_conditions,axis=-1)==1),axis=-1) ##due to ttcr2& dy2
        
        #req_dy_cr1 =ak.any((mT(leppair.lep2,met)>30)& (abs(ll_cand.mass-91.18)<15) & (events.MET.sumEt>45)& (mT(ll_cand,met)<60) ,axis=-1) 
        #req_dy_cr2 =ak.any((mT(leppair.lep2,met)>30)& (events.MET.sumEt>45)& (mT(ll_cand,met)<60)&(ak.sum(jet_conditions,axis=-1)==0) ,axis=-1) 
        #req_top_cr1 =ak.any((mT(leppair.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass>120),axis=-1) 
        #req_top_cr2 =ak.any((mT(leppair.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ak.count(jet_conditions,axis=1)>=2),axis=-1) 
        # req_WW_cr = ak.any((mT(leppair.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass),axis=-1) 
        
        #print(higgs_cand.type)
        #print(ll_cand.type)
        
        # global already contains Vpt>50 as the lower bound
        # global alsohas higgs_cand.mass<250
        req_sr_Zll = ak.any((ll_cand.mass<105) & (higgs_cand.delta_phi(ll_cand)>2.5) & (deepflavcvsltag(leading)>0.225) & (deepflavcvsbtag(leading)>0.4),
                            axis=-1)
        req_sr_Zll_vpt_low  = req_global & req_sr_Zll & ak.any(ll_cand.pt<150, axis=-1)
        req_sr_Zll_vpt_high = req_global & req_sr_Zll & ak.any(ll_cand.pt>150, axis=-1)
        
        
        #selection.add('llmass',ak.to_numpy(req_llmass))
        selection.add('SR',ak.to_numpy(req_sr_Zll))
        #selection.add('SR1',ak.to_numpy(req_sr1))
        #selection.add('SR2',ak.to_numpy(req_sr2))
        selection.add('SR_2LL',ak.to_numpy(req_sr_Zll_vpt_low))
        selection.add('SR_2LH',ak.to_numpy(req_sr_Zll_vpt_high))
        #selection.add('top_CR1',ak.to_numpy(req_top_cr1))
        #selection.add('top_CR2',ak.to_numpy(req_top_cr2))
        #selection.add('DY_CR1',ak.to_numpy(req_dy_cr1))
        #selection.add('DY_CR2',ak.to_numpy(req_dy_cr2))
        # selection.add('WW_CR',ak.to_numpy(req_WW_cr))
        
        #eventflav_jet = jets
        #sel_jet =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(leppair.lep2)>0.4,axis=2)]


        #sel_jetflav =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(leppair.lep2)>0.4,axis=2)]
        # sel_jetflav = ak.mask(sel_jetflav,ak.num(good_leptons)>0)
        #sel_cjet_flav = ak.pad_none(sel_jetflav,1,axis=1)
        #sel_cjet_flav = sel_cjet_flav[:,0]
        '''
        sel_jetcsv = eventcsv_jet[(eventcsv_jet.pt > 20) & (abs(eventcsv_jet.eta) <= 2.4)&((eventcsv_jet.puId > 0)|(eventcsv_jet.pt>50)) &(eventcsv_jet.jetId>5)&ak.all(eventcsv_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(leppair.lep2)>0.4,axis=2)]
        sel_cjet_csv = ak.pad_none(sel_jetcsv,1,axis=1)
        sel_cjet_csv = sel_cjet_csv[:,0]
        '''
        # sel_jetpn =  eventpn_jet[(eventpn_jet.pt > 20) & (abs(eventpn_jet.eta) <= 2.4)&((eventpn_jet.puId > 0)|(eventpn_jet.pt>50)) &(eventpn_jet.jetId>5)&ak.all(eventpn_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(leppair.lep2)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        # sel_jetpn = ak.mask(sel_jetpn,ak.num(pair_4lep)>0)
        # sel_cjet_pn = ak.pad_none(sel_jetpn,1,axis=1)
        # sel_cjet_pn = sel_cjet_pn[:,0]

        
        # last selection (what would be done in shapemaker): Ak15JetPt < 300 GeV
        
        
        
        if 'DoubleEG' in dataset :
            output['cutflow'][dataset]['trigger'] += ak.sum(trigger_ee)
        elif 'DoubleMuon' in dataset :
            output['cutflow'][dataset]['trigger'] += ak.sum(trigger_mm)
            
            
        # Successively add another cut w.r.t. previous line, looks a bit like N-1 histograms
        output['cutflow'][dataset]['global selection'] += ak.sum(req_global)
        output['cutflow'][dataset]['signal region'] += ak.sum(req_sr_Zll)
        output['cutflow'][dataset]['selected jets'] +=ak.sum(req_sr_Zll & (ak.sum(jet_conditions,axis=1)>=2) )
        output['cutflow'][dataset]['all ee'] +=ak.sum(req_sr_Zll & (ak.sum(jet_conditions,axis=1)>=2) & (nele==2) & (trigger_ee))
        output['cutflow'][dataset]['all mumu'] +=ak.sum(req_sr_Zll & (ak.sum(jet_conditions,axis=1)>=2) & (nmu==2) & trigger_mm)
        
        #output['cutflow'][dataset]['all emu'] +=ak.sum(req_sr&req_global&(ak.sum(jet_conditions,axis=1)>0)&(nele==1)&(nmu==1)&trigger_em)
        # output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)

        # see comment above
        lepflav = ['ee','mumu']
        # Old (from H+c)
        #reg = ['SR','SR_2LL','SR_2LH','DY_CR1','DY_CR2','top_CR1','top_CR2']
        # That's what I want in the end, all the regions from above
        #reg = regions    
        # That's what I have right now:
        reg = ['SR','SR_2LL','SR_2LH']
        
        #### write into histograms (i.e. write output)
        for histname, h in output.items():
            for ch in lepflav:
                for r in reg:
                    #if ch == 'emu':
                    #    cut = selection.all('lepsel', 'global_selection', 'metfilter', 'lumi', r, ch, 'trigger_%s'%(ch))
                    #elif ch == 'ee' or ch == 'mumu' :
                    #cut = selection.all('lepsel','global_selection','metfilter','lumi',r,ch, 'trigger_%s'%(ch),'llmass')
                    cut = selection.all('lepsel','jetsel','global_selection','metfilter','lumi', r, ch, 'trigger_%s'%(ch))
                    llcut = ll_cand[cut]
                    llcut = llcut[:,0]

                    lep1cut = llcut.lep1
                    lep2cut = llcut.lep2
                    if not isRealData:
                        if ch=='ee':
                            lepsf=eleSFs(lep1cut,self._year,self._corr)*eleSFs(lep2cut,self._year,self._corr)
                        elif ch=='mumu':
                            lepsf=muSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr)
                            
                        # This would be emu channel, which does not exist in the VHcc Zll case
                        '''
                        else:
                            lepsf= np.where(lep1cut.lep_flav==11,
                                           eleSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr),
                                           1.) \
                                   * np.where(lep1cut.lep_flav==13,
                                           eleSFs(lep2cut,self._year,self._corr)*muSFs(lep1cut,self._year,self._corr),
                                           1.)
                       '''
                    else : 
                        lepsf = weights.weight()[cut]
                    # print(weights.weight()[cut]*lepsf)
                    # print(lepsf)
                    if 'jetflav_' in histname:
                        fields = {l: normalize(leading[histname.replace('jetflav_','')],cut) for l in h.fields if l in dir(leading)}
                        if isRealData:
                            flavor= ak.zeros_like(normalize(leading['pt'],cut))
                        else :
                            flavor= normalize(leading.hadronFlavour+1*((leading.partonFlavour == 0 ) & (leading.hadronFlavour==0)),cut)
                        h.fill(dataset=dataset, lepflav =ch, region = r, flav=flavor, **fields,weight=weights.weight()[cut]*lepsf)  
                        '''
                        elif 'jetcsv_' in histname:
                            fields = {l: normalize(sel_cjet_csv[histname.replace('jetcsv_','')],cut) for l in h.fields if l in dir(sel_cjet_csv)}
                            h.fill(dataset=dataset,lepflav =ch, flav=normalize(sel_cjet_csv.hadronFlavour+1*((sel_cjet_csv.partonFlavour == 0 ) \
                            & (sel_cjet_csv.hadronFlavour==0)),cut), **fields,weight=weights.weight()[cut]*lepsf)    
                        '''
                    elif 'lep1_' in histname:
                        fields = {l: ak.fill_none(flatten(lep1cut[histname.replace('lep1_','')]),np.nan) for l in h.fields if l in dir(lep1cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'lep2_' in histname:
                        fields = {l: ak.fill_none(flatten(lep2cut[histname.replace('lep2_','')]),np.nan) for l in h.fields if l in dir(lep2cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'MET_' in histname:
                        fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                        h.fill(dataset=dataset, lepflav =ch, region = r,**fields,weight=weights.weight()[cut]*lepsf) 
                    elif 'll_' in histname:
                        fields = {l: ak.fill_none(flatten(llcut[histname.replace('ll_','')]),np.nan) for l in h.fields if l in dir(llcut)}
                        h.fill(dataset=dataset,lepflav=ch, region = r,**fields,weight=weights.weight()[cut]*lepsf) 
                    else :
                        output['nj'].fill(dataset=dataset,lepflav=ch,region = r,nj=normalize(ak.num(jet_conditions),cut),weight=weights.weight()[cut]*lepsf)                            
                        # print(ak.type(ak.flatten(mT(lep1cut,met[cut]))),ak.type(weights.weight()[cut]*lepsf))            
                        output['mT1'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(lep1cut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['mT2'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(lep2cut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['mTh'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(llcut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['dphi_ll'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(llcut)),weight=weights.weight()[cut]*lepsf)
                        output['dphi_lep1'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(lep1cut)),weight=weights.weight()[cut]*lepsf)
                        output['dphi_lep2'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(lep2cut)),weight=weights.weight()[cut]*lepsf)
                    
        return output

    def postprocess(self, accumulator):
        #print(accumulator)
        return accumulator
