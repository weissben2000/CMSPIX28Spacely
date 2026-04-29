# spacely
from Master_Config import *

# python modules
import sys
try:
    import time
    import tqdm
    from datetime import datetime
    import csv
except ImportError as e:
    loud_message(header_import_error, f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately

#-----------------------------------------------------------------------------
# ProgRead function
#-----------------------------------------------------------------------------
# This subroutine will take in a pattern, program the pixels via 
# setting pixelconfigs and injecting charge, then read the pixels back out
# and save them to a csv
#-----------------------------------------------------------------------------

def ProgRead():
    return