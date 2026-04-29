

INSTR = {"car" : {"type": "Caribou",
                  "host":"192.168.150.7",
                  "port":12345,
                  "device":"SpacelyCaribouBasic"}}



V_SEQUENCE = ["vdda", 
              "vddd", 
              "vth0", 
              "vth1",
              "vth2", 
              "VMC", 
              "SUPERPIX", 
              "INJ_1",
              "Ibias",
              "vdisc0",
              "vdisc1"

]

I_SEQUENCE = [
        #       "Ileak", 
            #   "OUTsink",
            #   "ThresholdOut",
            #   "ThresholdDown",
            #   "ThresholdUp",
            #   "Source1mA",
            #   "VlogSink1mA"
              ]

V_INSTR = {"vdda": "car",
           "vddd": "car",
           "vth0":"car",
           "vth1":"car",
           "vth2":"car",
           "VMC":"car",
           "SUPERPIX":"car",
           "INJ_1": "car",
           "Ibias": "car",
           "vdisc0" : "car",
           "vdisc1" : "car"
}

I_INSTR = {
        #    "Ileak": "car",
        #    "OUTsink": "car",
        #    "ThresholdOut": "car",
        #    "ThresholdDown": "car",
        #    "ThresholdUp": "car",
        #    "Source1mA": "car",
        #    "VlogSink1mA": "car"
           }

V_CHAN = {"vdda": "PWR_OUT_1",
           "vddd": "PWR_OUT_2",
           "vth0":"BIAS_4",
           "vth1":"BIAS_3",
           "vth2":"BIAS_2",
           "VMC":"BIAS_1",
           "SUPERPIX":"BIAS_5",
           "INJ_1":"INJ_1",
           "Ibias":"BIAS_26",
           "vdisc0":"BIAS_7",
           "vdisc1":"BIAS_9"
}

I_CHAN = {
        #    "Ileak":"CUR_8",
        #    "OUTsink":"CUR_1",
        #    "ThresholdOut":"CUR_2",
        #    "ThresholdDown":"CUR_3",
        #    "ThresholdUp":"CUR_4",
        #    "Source1mA":"CUR_5",
        #    "VlogSink1mA":"CUR_6"
           }

V_LEVEL = {"vdda": 0.9,
           "vddd": 0.9,
           "vth0": 0.031, #0.05 is 1000e-
           "vth1": 0.05, #0.08 is 1500e-
           "vth2": 0.11, #0.11 is 2000e-
           "VMC": 0.4,
           "SUPERPIX":0,
           "INJ_1": 2,
           "Ibias": 0.6,            #TUNE TO have 5uW/pixel
           "vdisc0" : 0.0,
           "vdisc1" : 0.0
}

I_LEVEL = {
        #    "Ileak": 0.01,  # 10uA
        #    "OUTsink": 0,
        #    "ThresholdOut": 0,
        #    "ThresholdDown": 0,
        #    "ThresholdUp": 0,
        #    "Source1mA": 0,
        #    "VlogSink1mA": 0
           }

V_WARN_VOLTAGE = {"vdda": [0.82,0.99],
           "vddd": [0.82,0.99],
           "vth0": [0,0.4],
           "vth1": [0,0.4],
           "vth2": [0,0.4],
           "VMC": [0,0.4],
           "SUPERPIX":[0,0.99],
           "INJ_1": [1.8,2.2],
           "Ibias": [0,0.9],
           "vdisc0": [0, 3.3],
           "vdisc1": [0, 3.3]
      }

V_PORT  = {"vdda": None,
           "vddd": None,
           "vth0": None,
           "vth1": None,
           "vth2": None,                      
           "VMC":None,
           "SUPERPIX":None,
           "INJ_1": None,
           "Ibias": None,
           "vdisc0": None,
           "vdisc1": None
}

I_PORT = {
        #    "Ileak": None,
        #    "OUTsink": None,
        #    "ThresholdOut": None,
        #    "ThresholdDown": None,
        #    "ThresholdUp": None,
        #    "Source1mA": None,
        #    "VlogSink1mA": None
           }

I_VOLT_LIMIT = {
            #   "Ileak": 0.01,
            #   "OUTsink": 0.01,
            #   "ThresholdOut": 0.01,
            #   "ThresholdDown": 0.01,
            #   "ThresholdUp": 0.01,
            #   "Source1mA": 0.01,
            #   "VlogSink1mA": 0.01
    }


FNAL_SETTINGS = {
    "storageDirectory" :"/nfs/cms/smartpix/CMSPIX28_DAQ/Results", #"/mnt/local/CMSPIX28/data",#"/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/Pnoise",#"/mnt/local/CMSPIX28/Scurve/data",#
    "chipVersion" : 1,
    "chipID" : 25
}

MP65_SPECIFIC = {
        "debug_csv"     : "/nfs/ceph/public/cms/smartpix/CMSPIX28_DAQ/spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely/csv/hidden_debug.csv",
        "weights_csv"   : "/nfs/ceph/public/cms/smartpix/CMSPIX28_DAQ/spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely/csv/b5_w5_b2_w2_pixel_bin.csv",
        "compout_csv"  : "/nfs/ceph/public/cms/smartpix/CMSPIX28_DAQ/spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely/csv/compouts.csv",
        "qkeras_model_file": "",
        "qkeras_model_pipeline_dir": ""
}


