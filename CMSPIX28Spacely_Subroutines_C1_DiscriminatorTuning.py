# spacely
from Master_Config import *

# python modules
import sys
import importlib.util
from pathlib import Path
import numpy as np
import csv
import os


def _get_qdense_batchnorm_fallback():
    """
    Local compatibility implementation based on qkeras PR #74.
    Used only when the installed qkeras package does not expose QDenseBatchnorm.
    """
    from qkeras.qlayers import QDense
    from tensorflow.keras import layers
    import tensorflow as tf

    class QDenseBatchnorm(QDense):
        def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            kernel_quantizer=None,
            bias_quantizer=None,
            kernel_range=None,
            bias_range=None,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            renorm=False,
            renorm_clipping=None,
            renorm_momentum=0.99,
            fused=None,
            trainable=True,
            virtual_batch_size=None,
            adjustment=None,
            ema_freeze_delay=None,
            folding_mode="ema_stats_folding",
            **kwargs,
        ):
            super().__init__(
                units=units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                kernel_quantizer=kernel_quantizer,
                bias_quantizer=bias_quantizer,
                kernel_range=kernel_range,
                bias_range=bias_range,
                **kwargs,
            )
            self.batchnorm = layers.BatchNormalization(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer,
                moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint,
                gamma_constraint=gamma_constraint,
                renorm=renorm,
                renorm_clipping=renorm_clipping,
                renorm_momentum=renorm_momentum,
                fused=fused,
                trainable=trainable,
                virtual_batch_size=virtual_batch_size,
                adjustment=adjustment,
            )
            self.ema_freeze_delay = ema_freeze_delay
            self.folding_mode = folding_mode

        def build(self, input_shape):
            super().build(input_shape)
            # Keep compatibility with checkpoints that store this variable.
            self._iteration = tf.Variable(
                -1,
                trainable=False,
                name="iteration",
                dtype=tf.int64,
            )

        def call(self, inputs, training=None):
            dense_out = tf.keras.backend.dot(inputs, self.kernel)
            if self.use_bias:
                dense_out = tf.keras.backend.bias_add(
                    dense_out, self.bias, data_format="channels_last"
                )
            out = self.batchnorm(dense_out, training=training)
            if self.activation is not None:
                out = self.activation(out)
            return out

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "ema_freeze_delay": self.ema_freeze_delay,
                "folding_mode": self.folding_mode,
            })
            # merge in BN config to support deserialization from saved models
            cfg.update(self.batchnorm.get_config())
            return cfg

    return QDenseBatchnorm


def _make_qkeras_custom_objects():
    import qkeras
    import qkeras.utils as qutils

    custom_objects = {}
    qutils._add_supported_quantized_objects(custom_objects)

    qdense_bn = custom_objects.get("QDenseBatchnorm")
    if qdense_bn is None:
        qdense_bn = _get_qdense_batchnorm_fallback()
        custom_objects["QDenseBatchnorm"] = qdense_bn

    # filter/model.py uses `from qkeras import *`; make sure symbol exists there.
    if not hasattr(qkeras, "QDenseBatchnorm"):
        setattr(qkeras, "QDenseBatchnorm", qdense_bn)

    return custom_objects


def _load_qkeras_model_module(model_pipeline_dir):
    # Ensure QDenseBatchnorm is available before importing filter/model.py,
    # since that file does `from qkeras import *` at import time.
    _make_qkeras_custom_objects()

    model_pipeline_dir = Path(model_pipeline_dir).resolve()
    model_py = model_pipeline_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Could not find model.py in {model_pipeline_dir}")

    # filter/model_pipeline uses bare imports and expects pretrain-data-prep.
    pretrain_data_prep_dir = (model_pipeline_dir.parent / "pretrain-data-prep").resolve()
    dataset_utils_py = pretrain_data_prep_dir / "dataset_utils.py"
    if not dataset_utils_py.exists():
        raise FileNotFoundError(
            "Could not find dataset_utils.py at "
            f"{dataset_utils_py}. Ensure filter submodules are initialized "
            "and model_pipeline_dir points to filter/model_pipeline."
        )

    for module_dir in (model_pipeline_dir, pretrain_data_prep_dir):
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

    spec = importlib.util.spec_from_file_location("smartpix_filter_model", str(model_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_qkeras_inference(yprofiles, qkeras_model_file, model_pipeline_dir, batch_size=2048):
    yprofiles = np.asarray(yprofiles, dtype=np.int32)
    if yprofiles.ndim != 2 or yprofiles.shape[1] != 16:
        raise ValueError(f"Expected yprofiles shape (N,16), got {yprofiles.shape}")

    md = _load_qkeras_model_module(model_pipeline_dir)
    try:
        qmodel = md.CreateQModel(shape=(16,), model_file=qkeras_model_file)
    except Exception as e:
        emsg = str(e)
        # Follow filter repo architecture path for weights-only files or
        # deserialization mismatches: build model then load weights.
        if (
            "No model config found" in emsg
            or "no model config found" in emsg
            or ("expects" in emsg and "weights" in emsg)
            or "QDenseBatchnorm" in emsg
        ):
            qmodel = md.CreateQModel(shape=(16,), model_file=None)
            qmodel.load_weights(qkeras_model_file)
        else:
            raise
    logits = qmodel.predict(yprofiles, batch_size=batch_size, verbose=0)
    predictions = np.argmax(logits, axis=1).astype(np.int32)
    return {"logits": logits, "predictions": predictions}


def decode_asic_readouts(readouts, latency_bit=37, bxclkFreq="28"):
    fdivider = int(bxclkFreq, 16)
    dnn_s = [''.join(row) if isinstance(row, (list, tuple)) else str(row) for row in readouts]
    dnn_0 = [item[-fdivider:] for item in dnn_s]
    dnn_1 = [item[-(64 + fdivider):-64] for item in dnn_s]
    bxclk_ana = [item[-(128 + fdivider):-128] for item in dnn_s]

    # keep same behavior as DNN_analyse: no latency shift unless explicitly debugging there
    decoded = []
    for tv in range(len(dnn_0)):
        cnt_dnn0_ones = 0
        cnt_dnn1_zeros = 0
        for index, _ in enumerate(str(bxclk_ana[tv])):
            if dnn_0[tv][index] == "1":
                cnt_dnn0_ones += 1
            if dnn_1[tv][index] == "0":
                cnt_dnn1_zeros += 1
        bit0 = 1 if cnt_dnn0_ones > 4 else 0
        bit1 = 0 if cnt_dnn1_zeros > 4 else 1
        decoded.append((bit1 << 1) | bit0)
    return np.array(decoded, dtype=np.int32)


def compare_asic_to_qkeras_bits(asic_codes, qkeras_classes, qkeras_to_asic_map=None):
    if qkeras_to_asic_map is None:
        qkeras_to_asic_map = {0: 0, 1: 1, 2: 2}

    asic_codes = np.asarray(asic_codes, dtype=np.int32)
    qkeras_classes = np.asarray(qkeras_classes, dtype=np.int32)
    if len(asic_codes) != len(qkeras_classes):
        raise ValueError("ASIC and QKeras outputs must have identical lengths")

    mapped_qkeras = np.array([qkeras_to_asic_map[int(c)] for c in qkeras_classes], dtype=np.int32)
    asic_bit0 = asic_codes & 0b01
    asic_bit1 = (asic_codes >> 1) & 0b01
    q_bit0 = mapped_qkeras & 0b01
    q_bit1 = (mapped_qkeras >> 1) & 0b01

    bit0_err = (asic_bit0 != q_bit0).astype(np.int32)
    bit1_err = (asic_bit1 != q_bit1).astype(np.int32)
    class_err = (asic_codes != mapped_qkeras).astype(np.int32)
    return {
        "mapped_qkeras_codes": mapped_qkeras,
        "bit0_error_rate": float(np.mean(bit0_err)) if len(bit0_err) else 0.0,
        "bit1_error_rate": float(np.mean(bit1_err)) if len(bit1_err) else 0.0,
        "class_error_rate": float(np.mean(class_err)) if len(class_err) else 0.0,
        "n": int(len(class_err)),
    }


def optimize_discriminator_thresholds(
    qkeras_model_file=None,
    model_pipeline_dir=None,
    patternIndexes=None,
    n_test_vectors=1,
    init_vdisc0=0.0,
    init_vdisc1=0.0,
    step_vdisc0=0.01,
    step_vdisc1=0.01,
    max_iters=8,
    qkeras_to_asic_map=None,
    vmin=0.0,
    vmax=3.3,
    dnn_kwargs=None,
):
    """
    Tune discriminator thresholds independently:
      - vdisc0 optimized using bit0 mismatch
      - vdisc1 optimized using bit1 mismatch
    """
    if dnn_kwargs is None:
        dnn_kwargs = {}
    if qkeras_model_file is None:
        qkeras_model_file = MP65_SPECIFIC.get("qkeras_model_file")
    if model_pipeline_dir is None:
        model_pipeline_dir = MP65_SPECIFIC.get("qkeras_model_pipeline_dir")
    if not qkeras_model_file or not model_pipeline_dir:
        raise ValueError(
            "QKeras paths are missing. Set MP65_SPECIFIC['qkeras_model_file'] "
            "and MP65_SPECIFIC['qkeras_model_pipeline_dir'], or pass them as args."
        )
    if patternIndexes is None:
        n_test_vectors = int(n_test_vectors)
        if n_test_vectors <= 0:
            raise ValueError("n_test_vectors must be > 0")
        patternIndexes = list(range(n_test_vectors))

    def _clamp_vdisc(v, name):
        v_clamped = float(np.clip(float(v), 0.0, 3.3))
        if v_clamped != float(v):
            print(f"[SAFETY] Clamped {name} from {v} to {v_clamped} V")
        return v_clamped

    vdisc0 = _clamp_vdisc(init_vdisc0, "vdisc0")
    vdisc1 = _clamp_vdisc(init_vdisc1, "vdisc1")
    vmin = max(0.0, float(vmin))
    vmax = min(3.3, float(vmax))
    if vmin > vmax:
        raise ValueError("Voltage guard limits invalid after applying [0, 3.3] bounds")

    def evaluate(v0, v1):
        v0 = _clamp_vdisc(v0, "vdisc0")
        v1 = _clamp_vdisc(v1, "vdisc1")
        V_PORT["vdisc0"].set_voltage(v0)
        V_LEVEL["vdisc0"] = v0
        V_PORT["vdisc1"].set_voltage(v1)
        V_LEVEL["vdisc1"] = v1

        dnn_result = DNN(
            patternIndexes=patternIndexes,
            return_data=True,
            readYproj=True,
            **dnn_kwargs,
        )
        yprofiles = dnn_result["yprofiles"]
        if yprofiles is None:
            raise RuntimeError("DNN returned no yprofiles.")

        qres = run_qkeras_inference(
            yprofiles=yprofiles,
            qkeras_model_file=qkeras_model_file,
            model_pipeline_dir=model_pipeline_dir,
        )
        asic_codes = decode_asic_readouts(dnn_result["readouts"])
        cmp_res = compare_asic_to_qkeras_bits(
            asic_codes=asic_codes,
            qkeras_classes=qres["predictions"],
            qkeras_to_asic_map=qkeras_to_asic_map,
        )
        return cmp_res, dnn_result["outDir"]

    cmp_res, outdir = evaluate(vdisc0, vdisc1)
    best_bit0 = cmp_res["bit0_error_rate"]
    best_bit1 = cmp_res["bit1_error_rate"]
    history = [{
        "iter": 0,
        "vdisc0": vdisc0,
        "vdisc1": vdisc1,
        "bit0_error_rate": best_bit0,
        "bit1_error_rate": best_bit1,
        "class_error_rate": cmp_res["class_error_rate"],
        "outDir": outdir,
    }]

    for it in range(1, max_iters + 1):
        improved = False

        # Tune vdisc0 for bit0 only.
        v0_candidates = []
        for sign in (-1.0, 1.0):
            trial_v0 = float(np.clip(vdisc0 + sign * step_vdisc0, vmin, vmax))
            trial_cmp, trial_outdir = evaluate(trial_v0, vdisc1)
            v0_candidates.append((trial_cmp["bit0_error_rate"], trial_v0, trial_cmp, trial_outdir))
        cand_bit0, cand_v0, cand_cmp0, cand_outdir0 = min(v0_candidates, key=lambda x: x[0])
        if cand_bit0 < best_bit0:
            vdisc0 = cand_v0
            best_bit0 = cand_bit0
            cmp_res = cand_cmp0
            outdir = cand_outdir0
            improved = True

        # Tune vdisc1 for bit1 only.
        v1_candidates = []
        for sign in (-1.0, 1.0):
            trial_v1 = float(np.clip(vdisc1 + sign * step_vdisc1, vmin, vmax))
            trial_cmp, trial_outdir = evaluate(vdisc0, trial_v1)
            v1_candidates.append((trial_cmp["bit1_error_rate"], trial_v1, trial_cmp, trial_outdir))
        cand_bit1, cand_v1, cand_cmp1, cand_outdir1 = min(v1_candidates, key=lambda x: x[0])
        if cand_bit1 < best_bit1:
            vdisc1 = cand_v1
            best_bit1 = cand_bit1
            cmp_res = cand_cmp1
            outdir = cand_outdir1
            improved = True

        history.append({
            "iter": it,
            "vdisc0": vdisc0,
            "vdisc1": vdisc1,
            "bit0_error_rate": best_bit0,
            "bit1_error_rate": best_bit1,
            "class_error_rate": cmp_res["class_error_rate"],
            "outDir": outdir,
        })
        if not improved:
            break

    summary = {
        "best_discriminators": {"vdisc0": vdisc0, "vdisc1": vdisc1},
        "best_metrics": {
            "bit0_error_rate": best_bit0,
            "bit1_error_rate": best_bit1,
            "class_error_rate": cmp_res["class_error_rate"],
        },
        "history": history,
    }

    history_csv = os.path.join(history[-1]["outDir"], "discriminator_tuning_history.csv")
    csv_columns = ["iter", "vdisc0", "vdisc1", "bit0_error_rate", "bit1_error_rate", "class_error_rate", "outDir"]
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(history)
    summary["history_csv"] = history_csv
    return summary


def optimize_discriminator_thresholds_experimental(
    qkeras_model_file=None,
    model_pipeline_dir=None,
    patternIndexes=None,
    n_test_vectors=1,
    init_vdisc0=0.0,
    init_vdisc1=0.0,
    step_vdisc0=0.01,
    step_vdisc1=0.01,
    max_iters=8,
    qkeras_to_asic_map=None,
    vmin=0.0,
    vmax=3.3,
    dnn_kwargs=None,
    step_grow=1.25,
    step_shrink=0.5,
    min_step=0.001,
    max_step=0.2,
):
    """
    Experimental optimizer:
      - same independent bit objective as baseline
      - dynamically adapts step size per bit from observed improvement
    """
    if dnn_kwargs is None:
        dnn_kwargs = {}
    if qkeras_model_file is None:
        qkeras_model_file = MP65_SPECIFIC.get("qkeras_model_file")
    if model_pipeline_dir is None:
        model_pipeline_dir = MP65_SPECIFIC.get("qkeras_model_pipeline_dir")
    if not qkeras_model_file or not model_pipeline_dir:
        raise ValueError(
            "QKeras paths are missing. Set MP65_SPECIFIC['qkeras_model_file'] "
            "and MP65_SPECIFIC['qkeras_model_pipeline_dir'], or pass them as args."
        )
    if patternIndexes is None:
        n_test_vectors = int(n_test_vectors)
        if n_test_vectors <= 0:
            raise ValueError("n_test_vectors must be > 0")
        patternIndexes = list(range(n_test_vectors))

    def _clamp_vdisc(v, name):
        v_clamped = float(np.clip(float(v), 0.0, 3.3))
        if v_clamped != float(v):
            print(f"[SAFETY] Clamped {name} from {v} to {v_clamped} V")
        return v_clamped

    vdisc0 = _clamp_vdisc(init_vdisc0, "vdisc0")
    vdisc1 = _clamp_vdisc(init_vdisc1, "vdisc1")
    vmin = max(0.0, float(vmin))
    vmax = min(3.3, float(vmax))
    if vmin > vmax:
        raise ValueError("Voltage guard limits invalid after applying [0, 3.3] bounds")

    s0 = float(np.clip(step_vdisc0, min_step, max_step))
    s1 = float(np.clip(step_vdisc1, min_step, max_step))

    def evaluate(v0, v1):
        v0 = _clamp_vdisc(v0, "vdisc0")
        v1 = _clamp_vdisc(v1, "vdisc1")
        V_PORT["vdisc0"].set_voltage(v0)
        V_LEVEL["vdisc0"] = v0
        V_PORT["vdisc1"].set_voltage(v1)
        V_LEVEL["vdisc1"] = v1

        dnn_result = DNN(
            patternIndexes=patternIndexes,
            return_data=True,
            readYproj=True,
            **dnn_kwargs,
        )
        yprofiles = dnn_result["yprofiles"]
        if yprofiles is None:
            raise RuntimeError("DNN returned no yprofiles.")

        qres = run_qkeras_inference(
            yprofiles=yprofiles,
            qkeras_model_file=qkeras_model_file,
            model_pipeline_dir=model_pipeline_dir,
        )
        asic_codes = decode_asic_readouts(dnn_result["readouts"])
        cmp_res = compare_asic_to_qkeras_bits(
            asic_codes=asic_codes,
            qkeras_classes=qres["predictions"],
            qkeras_to_asic_map=qkeras_to_asic_map,
        )
        return cmp_res, dnn_result["outDir"]

    cmp_res, outdir = evaluate(vdisc0, vdisc1)
    best_bit0 = cmp_res["bit0_error_rate"]
    best_bit1 = cmp_res["bit1_error_rate"]
    history = [{
        "iter": 0,
        "vdisc0": vdisc0,
        "vdisc1": vdisc1,
        "step_vdisc0": s0,
        "step_vdisc1": s1,
        "bit0_error_rate": best_bit0,
        "bit1_error_rate": best_bit1,
        "class_error_rate": cmp_res["class_error_rate"],
        "outDir": outdir,
    }]

    for it in range(1, max_iters + 1):
        improved = False

        v0_candidates = []
        for sign in (-1.0, 1.0):
            trial_v0 = float(np.clip(vdisc0 + sign * s0, vmin, vmax))
            trial_cmp, trial_outdir = evaluate(trial_v0, vdisc1)
            v0_candidates.append((trial_cmp["bit0_error_rate"], trial_v0, trial_cmp, trial_outdir))
        cand_bit0, cand_v0, cand_cmp0, cand_outdir0 = min(v0_candidates, key=lambda x: x[0])
        if cand_bit0 < best_bit0:
            vdisc0 = cand_v0
            best_bit0 = cand_bit0
            cmp_res = cand_cmp0
            outdir = cand_outdir0
            s0 = float(min(max_step, s0 * step_grow))
            improved = True
        else:
            s0 = float(max(min_step, s0 * step_shrink))

        v1_candidates = []
        for sign in (-1.0, 1.0):
            trial_v1 = float(np.clip(vdisc1 + sign * s1, vmin, vmax))
            trial_cmp, trial_outdir = evaluate(vdisc0, trial_v1)
            v1_candidates.append((trial_cmp["bit1_error_rate"], trial_v1, trial_cmp, trial_outdir))
        cand_bit1, cand_v1, cand_cmp1, cand_outdir1 = min(v1_candidates, key=lambda x: x[0])
        if cand_bit1 < best_bit1:
            vdisc1 = cand_v1
            best_bit1 = cand_bit1
            cmp_res = cand_cmp1
            outdir = cand_outdir1
            s1 = float(min(max_step, s1 * step_grow))
            improved = True
        else:
            s1 = float(max(min_step, s1 * step_shrink))

        history.append({
            "iter": it,
            "vdisc0": vdisc0,
            "vdisc1": vdisc1,
            "step_vdisc0": s0,
            "step_vdisc1": s1,
            "bit0_error_rate": best_bit0,
            "bit1_error_rate": best_bit1,
            "class_error_rate": cmp_res["class_error_rate"],
            "outDir": outdir,
        })
        if not improved and s0 <= min_step and s1 <= min_step:
            break

    summary = {
        "best_discriminators": {"vdisc0": vdisc0, "vdisc1": vdisc1},
        "best_metrics": {
            "bit0_error_rate": best_bit0,
            "bit1_error_rate": best_bit1,
            "class_error_rate": cmp_res["class_error_rate"],
        },
        "history": history,
    }

    history_csv = os.path.join(history[-1]["outDir"], "discriminator_tuning_history_experimental.csv")
    csv_columns = [
        "iter", "vdisc0", "vdisc1", "step_vdisc0", "step_vdisc1",
        "bit0_error_rate", "bit1_error_rate", "class_error_rate", "outDir"
    ]
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(history)
    summary["history_csv"] = history_csv
    return summary
