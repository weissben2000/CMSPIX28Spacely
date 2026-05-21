from Master_Config import *

import sys
try:
    import os
    import time
    import csv
    from datetime import datetime
except ImportError as e:
    loud_message(header_import_error, f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately


THERMO_TRIPLET_TO_VALUE = {
    "000": 0,
    "001": 1,
    "011": 2,
    "111": 3,
}

_ARRAY_SIZE = 16
_SCANCHAIN_BITS = 768


def _pixel_num_from_16x16(row_16, col_16):
    row_8 = row_16 // 2
    half = row_16 % 2
    col_8 = half * 16 + col_16
    return grid[7 - row_8][col_8]


def _rc16_from_pixel_num(pixel_num):
    for row_8 in range(8):
        for col_8 in range(32):
            if grid[7 - row_8][col_8] == pixel_num:
                return row_8 * 2 + (col_8 // 16), col_8 % 16
    raise ValueError(f"pixel number {pixel_num} not found in grid")


def load_16x16_csv(csv_path):
    with open(csv_path, newline="") as file:
        rows = list(csv.reader(file))
    rows = [[c.strip() for c in row if c.strip() != ""] for row in rows if any(c.strip() for c in row)]
    if len(rows) != _ARRAY_SIZE or any(len(row) != _ARRAY_SIZE for row in rows):
        raise ValueError(
            f"expected {_ARRAY_SIZE}x{_ARRAY_SIZE} CSV at {csv_path}, got {len(rows)} rows"
        )
    arr = []
    for row in rows:
        line = []
        for cell in row:
            val = int(cell)
            if val not in (0, 1, 2, 3):
                raise ValueError(f"pixel value must be 0-3, got {val} in {csv_path}")
            line.append(val)
        arr.append(line)
    return arr


def save_16x16_csv(csv_path, arr):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(arr)


def sixteen_by_sixteen_to_pixel_lists(arr):
    pixel_list = []
    pixel_values = []
    for row_16 in range(_ARRAY_SIZE):
        for col_16 in range(_ARRAY_SIZE):
            pixel_list.append(_pixel_num_from_16x16(row_16, col_16))
            pixel_values.append(int(arr[row_16][col_16]))
    return pixel_list, pixel_values


def decode_thermo_triplet(triplet):
    key = "".join(triplet)
    if key not in THERMO_TRIPLET_TO_VALUE:
        raise ValueError(f"invalid thermometric readout code '{key}'")
    return THERMO_TRIPLET_TO_VALUE[key]


def scan_bits_to_16x16(bit_string):
    if len(bit_string) != _SCANCHAIN_BITS:
        raise ValueError(
            f"expected {_SCANCHAIN_BITS} scanchain bits, got {len(bit_string)}"
        )
    arr = [[0] * _ARRAY_SIZE for _ in range(_ARRAY_SIZE)]
    for pixel_num in range(256):
        triplet = bit_string[pixel_num * 3 : (pixel_num + 1) * 3]
        row_16, col_16 = _rc16_from_pixel_num(pixel_num)
        arr[row_16][col_16] = decode_thermo_triplet(triplet)
    return arr


def _mean_per_cell_diff(original, readouts):
    n = len(readouts)
    mean = [[0.0] * _ARRAY_SIZE for _ in range(_ARRAY_SIZE)]
    for row in range(_ARRAY_SIZE):
        for col in range(_ARRAY_SIZE):
            mean[row][col] = sum(
                readouts[i][row][col] - original[row][col] for i in range(n)
            ) / n
    return mean


def ProgImage(
    csv_path,
    configclk_period="64",
    cfg_test_delay="14",
    cfg_test_sample="0F",
    cfg_test_gate_config_clk="1",
):
    arr = load_16x16_csv(csv_path)
    pixel_list, pixel_values = sixteen_by_sixteen_to_pixel_lists(arr)
    ProgPixelsOnly(
        configclk_period=configclk_period,
        cfg_test_delay=cfg_test_delay,
        cfg_test_sample=cfg_test_sample,
        cfg_test_gate_config_clk=cfg_test_gate_config_clk,
        pixelList=pixel_list,
        pixelValue=pixel_values,
    )


def ProgRead(
    csv_path,
    configclk_period="64",
    cfg_test_delay="14",
    cfg_test_sample="0F",
    cfg_test_gate_config_clk="1",
    scan_load_delay="13",
    startBxclkState="0",
    bxclk_delay="0B",
    bxclk_period="28",
    injection_delay="1D",
    scanLoopBackBit="0",
    test_sample="08",
    test_delay="03",
    scanLoadPhase="20",
    post_prog_delay_s=0.5,
    n_cycles=1,
    run_label=None,
    results_base_dir=None,
):
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")

    original = load_16x16_csv(csv_path)
    base_dir = results_base_dir or MP65_SPECIFIC["progread_results_dir"]
    stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    label = run_label or os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(base_dir, f"{stamp}_{label}")
    os.makedirs(out_dir, exist_ok=True)

    save_16x16_csv(os.path.join(out_dir, "original.csv"), original)

    readouts = []
    for attempt in range(1, n_cycles + 1):
        ProgImage(
            csv_path,
            configclk_period=configclk_period,
            cfg_test_delay=cfg_test_delay,
            cfg_test_sample=cfg_test_sample,
            cfg_test_gate_config_clk=cfg_test_gate_config_clk,
        )
        time.sleep(post_prog_delay_s)
        bit_string = ScanChainOneShot(
            scan_load_delay=scan_load_delay,
            startBxclkState=startBxclkState,
            bxclk_delay=bxclk_delay,
            bxclk_period=bxclk_period,
            injection_delay=injection_delay,
            scanLoopBackBit=scanLoopBackBit,
            test_sample=test_sample,
            test_delay=test_delay,
            scanLoadPhase=scanLoadPhase,
        )
        readout = scan_bits_to_16x16(bit_string)
        readouts.append(readout)
        save_16x16_csv(
            os.path.join(out_dir, f"readout_attempt_{attempt:03d}.csv"),
            readout,
        )

    mean_diff = _mean_per_cell_diff(original, readouts)
    with open(os.path.join(out_dir, "average_difference.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(mean_diff)

    return {
        "output_dir": out_dir,
        "n_cycles": n_cycles,
        "readouts": readouts,
        "average_difference": mean_diff,
    }
