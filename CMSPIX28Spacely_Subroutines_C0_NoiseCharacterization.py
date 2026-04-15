# spacely
from Master_Config import *

# python modules
import sys
try:
    pass
except ImportError as e:
    loud_message(header_import_error, f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately

