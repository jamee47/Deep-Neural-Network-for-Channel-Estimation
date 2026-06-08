"""
Two-Stage CNN + LSTM Framework for DVB-S2 GEO Links with ACM Optimization
===========================================================================
Stage 1: Intra-Frame Pilot Processing (CNN) – pilots treated as blocks
Stage 2: Inter-Time Temporal Prediction (Bi-LSTM) with MODCOD features

Reference architecture from the thesis diagram:
  Block-Based Pilot Processing → Time-Series Prediction of Future Channel
  Quality → ACM via ETSI Lookup Table
"""

import tensorflow as tf
import keras
from keras import layers, Model, regularizers, ops


# ---------------------------------------------------------------------------
# Stage 1 – CNN Feature Extractor (shared across all blocks)
# ---------------------------------------------------------------------------
class CNNFeatureExtractor(layers.Layer):
    """Processes a single pilot block through a stack of Conv1D layers.

    Architecture (per the diagram):
        Conv1D(32) → BN + ReLU
        Conv1D(64) → BN + ReLU
        Conv1D(128) → BN + ReLU
        → Global Average Pooling
        → Dense(d_f)   # project to fixed feature dimension

    Parameters
    ----------
    d_f : int
        Output feature dimension per block (default 64).
    kernel_size : int
        Convolution kernel size (default 3).
    l2_reg : float
        L2 regularization factor applied to Conv1D kernels.
    """

    def __init__(self, d_f=64, kernel_size=3, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.d_f = d_f

        # Three Conv1D blocks: 32 → 64 → 128 filters
        self.conv1 = layers.Conv1D(
            32, kernel_size, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv1d_32",
        )
        self.bn1 = layers.BatchNormalization(name="bn_32")

        self.conv2 = layers.Conv1D(
            64, kernel_size, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv1d_64",
        )
        self.bn2 = layers.BatchNormalization(name="bn_64")

        self.conv3 = layers.Conv1D(
            128, kernel_size, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv1d_128",
        )
        self.bn3 = layers.BatchNormalization(name="bn_128")

        self.gap = layers.GlobalAveragePooling1D(name="global_avg_pool")
        self.dense_proj = layers.Dense(d_f, activation="relu", name="block_proj")

    def call(self, x, mask=None, training=False):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, N_max, 2)
            Pilot block input.  Each pilot has [Re(y), Im(y)] or
            [Re(y), Im(y), x, y] depending on configuration.
        mask : Tensor, shape (batch, N_max) or None
            Binary mask (1 = valid pilot, 0 = padded).
        training : bool
            Whether in training mode (affects BN / dropout).

        Returns
        -------
        Tensor, shape (batch, d_f)
            Block-level feature vector.
        """
        # Apply mask by zeroing out padded positions before convolutions
        if mask is not None:
            # mask: (batch, N_max) → (batch, N_max, 1)
            x = x * ops.expand_dims(ops.cast(mask, x.dtype), axis=-1)

        h = ops.relu(self.bn1(self.conv1(x), training=training))
        h = ops.relu(self.bn2(self.conv2(h), training=training))
        h = ops.relu(self.bn3(self.conv3(h), training=training))

        h = self.gap(h)                   # (batch, 128)
        h = self.dense_proj(h)            # (batch, d_f)
        return h

    def get_config(self):
        config = super().get_config()
        config.update({"d_f": self.d_f})
        return config


# ---------------------------------------------------------------------------
# Stage 1 – Per-Frame Block Aggregator
# ---------------------------------------------------------------------------
class PilotBlockAggregator(layers.Layer):
    """Groups frame pilots into B blocks, extracts CNN features per block,
    and aggregates them into a single per-frame feature vector s_k.

    Parameters
    ----------
    num_blocks : int
        Number of pilot blocks B per DVB-S2 frame (default 22).
        DVB-S2 standard: 792 pilots / 36 symbols per block = 22 blocks.
    n_max : int
        Number of pilot symbols per block (default 36, DVB-S2 standard).
    d_f : int
        Feature dimension output by the CNN per block (default 64).
    d_s : int
        Output per-frame feature size, i.e. the channel-state vector
        dimension (default 5).  Represents [|H|, A(dB), phase, SNR, noise_var].
    kernel_size : int
        CNN kernel size (default 3).
    l2_reg : float
        L2 weight decay (default 1e-4).
    """

    def __init__(
        self,
        num_blocks=22,
        n_max=36,
        d_f=64,
        d_s=5,
        kernel_size=3,
        l2_reg=1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.n_max = n_max
        self.d_f = d_f
        self.d_s = d_s

        # Shared CNN across all blocks
        self.cnn = CNNFeatureExtractor(
            d_f=d_f, kernel_size=kernel_size, l2_reg=l2_reg, name="cnn_extractor"
        )

        # Aggregation: concatenated block features → per-frame vector
        # Input dim = num_blocks * d_f, output dim = d_s
        self.agg_dense1 = layers.Dense(128, activation="relu", name="agg_dense1")
        self.agg_dense2 = layers.Dense(d_s, activation=None, name="agg_dense2")

    def call(self, blocks, masks=None, training=False):
        """
        Parameters
        ----------
        blocks : Tensor, shape (batch, B, N_max, C_pilot)
            Pilot values for each block.  C_pilot is the per-pilot feature
            dimension (e.g. 2 for [Re, Im]).
        masks : Tensor, shape (batch, B, N_max) or None
            Binary masks per block (1 = valid, 0 = pad).

        Returns
        -------
        s_k : Tensor, shape (batch, d_s)
            Per-frame channel-state feature vector.
        """
        block_features = []

        for b in range(self.num_blocks):
            block_input = blocks[:, b, :, :]          # (batch, N_max, C_pilot)
            block_mask = masks[:, b, :] if masks is not None else None
            f_b = self.cnn(block_input, mask=block_mask, training=training)
            block_features.append(f_b)                # each (batch, d_f)

        # Concatenate all block features → (batch, B * d_f)
        aggregated = ops.concatenate(block_features, axis=-1)

        # Project to per-frame feature vector
        s_k = self.agg_dense1(aggregated)             # (batch, 128)
        s_k = self.agg_dense2(s_k)                    # (batch, d_s)
        return s_k

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "n_max": self.n_max,
            "d_f": self.d_f,
            "d_s": self.d_s,
        })
        return config


# ---------------------------------------------------------------------------
# Stage 1 – Time-Distributed Wrapper
# ---------------------------------------------------------------------------
class Stage1TimeDistributed(layers.Layer):
    """Apply PilotBlockAggregator independently to each time step.

    Merges the batch and time dimensions, runs the aggregator once
    (efficient shared-weight application), then reshapes back.

    Parameters
    ----------
    aggregator : PilotBlockAggregator
        A pre-built Stage 1 aggregator layer.
    history_len : int
        Number of time steps T_h (needed for static reshape).
    """

    def __init__(self, aggregator, history_len, **kwargs):
        super().__init__(**kwargs)
        self.aggregator = aggregator
        self.history_len = history_len

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : list of two Tensors
            [pilot_blocks, pilot_masks]
            pilot_blocks : (batch, T_h, B, N_max, C_pilot)
            pilot_masks  : (batch, T_h, B, N_max)

        Returns
        -------
        channel_features : Tensor, shape (batch, T_h, d_s)
        """
        pilot_blocks, pilot_masks = inputs

        # Merge batch and time: (batch * T_h, B, N_max, C)
        shape_blocks = ops.shape(pilot_blocks)
        batch_size = shape_blocks[0]
        merged_blocks = ops.reshape(
            pilot_blocks,
            (batch_size * self.history_len,) + tuple(shape_blocks[2:])
        )
        shape_masks = ops.shape(pilot_masks)
        merged_masks = ops.reshape(
            pilot_masks,
            (batch_size * self.history_len,) + tuple(shape_masks[2:])
        )

        # Run aggregator on all (batch * T_h) frames at once
        features = self.aggregator(
            merged_blocks, merged_masks, training=training
        )  # (batch * T_h, d_s)

        # Reshape back: (batch, T_h, d_s)
        d_s = ops.shape(features)[-1]
        channel_features = ops.reshape(
            features, (batch_size, self.history_len, d_s)
        )
        return channel_features

    def get_config(self):
        config = super().get_config()
        config.update({"history_len": self.history_len})
        return config


# ---------------------------------------------------------------------------
# Stage 2 – Bi-LSTM Temporal Predictor
# ---------------------------------------------------------------------------
class BiLSTMPredictor(layers.Layer):
    """Bi-directional LSTM encoder followed by per-step Dense heads.

    Takes a time-series of input feature vectors (channel state + MODCOD
    context) and predicts SNR for the next H future steps.

    Architecture (per the diagram):
        Input  →  Bi-LSTM (n_layers)  →  Concatenation (2 × hidden)
        →  H independent Dense heads (one per future step)

    Parameters
    ----------
    hidden_units : int
        Number of LSTM units per direction (default 128).
    n_layers : int
        Number of stacked Bi-LSTM layers (default 2).
    prediction_horizon : int
        H – number of future time steps to predict (default 50).
    dropout_rate : float
        Recurrent + input dropout rate (default 0.1).
    l2_reg : float
        L2 regularization on Dense output heads (default 1e-4).
    """

    def __init__(
        self,
        hidden_units=128,
        n_layers=2,
        prediction_horizon=50,
        dropout_rate=0.1,
        l2_reg=1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.prediction_horizon = prediction_horizon

        # Stacked Bi-LSTM layers
        self.bilstm_layers = []
        for i in range(n_layers):
            lstm = layers.LSTM(
                hidden_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name=f"lstm_layer_{i}",
            )
            bilstm = layers.Bidirectional(lstm, name=f"bilstm_{i}")
            self.bilstm_layers.append(bilstm)

        # Per-step prediction heads – each outputs a single SNR value
        self.prediction_heads = []
        for h in range(prediction_horizon):
            head = layers.Dense(
                1,
                activation=None,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"snr_head_step_{h+1}",
            )
            self.prediction_heads.append(head)

    def call(self, x, training=False):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, T_h, d_in)
            Input time-series.  T_h = history window length,
            d_in = d_s + 2 (channel features + MODCOD features).

        Returns
        -------
        predictions : Tensor, shape (batch, H)
            Predicted SNR (dB) for each of the next H future time steps.
        """
        h = x
        for bilstm in self.bilstm_layers:
            h = bilstm(h, training=training)
        # h shape: (batch, T_h, 2 * hidden_units)

        # Use the last time-step encoding as context for prediction
        context = h[:, -1, :]  # (batch, 2 * hidden_units)

        # Each head predicts SNR for one future step
        preds = []
        for head in self.prediction_heads:
            p = head(context)           # (batch, 1)
            preds.append(p)

        predictions = ops.concatenate(preds, axis=-1)  # (batch, H)
        return predictions

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "n_layers": self.n_layers,
            "prediction_horizon": self.prediction_horizon,
        })
        return config


# ---------------------------------------------------------------------------
# Full Two-Stage Model (Functional API wrapper)
# ---------------------------------------------------------------------------
def build_two_stage_model(
    # Stage 1 params
    num_blocks: int = 22,
    n_max: int = 36,
    c_pilot: int = 2,
    d_f: int = 64,
    d_s: int = 5,
    cnn_kernel_size: int = 3,
    # Stage 2 params
    history_len: int = 200,
    prediction_horizon: int = 50,
    hidden_units: int = 128,
    n_lstm_layers: int = 2,
    dropout_rate: float = 0.1,
    # Regularization
    l2_reg: float = 1e-4,
    # Loss weights
    lambda_snr: float = 1.0,
    lambda_att: float = 0.0,
    lambda_fm: float = 0.0,
    learning_rate: float = 1e-3,
) -> Model:
    """Build and compile the full Two-Stage CNN + Bi-LSTM model.

    Stage 1 is applied independently at each time step in the history
    window to produce per-frame features.  Stage 2 consumes the resulting
    time-series (augmented with MODCOD context) and predicts future SNR.

    Parameters
    ----------
    num_blocks : int
        B – number of pilot blocks per frame (default 22).
        DVB-S2: 792 pilots / 36 per block = 22 blocks.
    n_max : int
        Pilots per block (default 36, DVB-S2 standard block size).
    c_pilot : int
        Channels per pilot sample, e.g. 2 for [Re, Im] (default 2).
    d_f : int
        CNN block feature dimension (default 64).
    d_s : int
        Per-frame channel-state feature dimension (default 5).
    cnn_kernel_size : int
        Kernel size for CNN (default 3).
    history_len : int
        T_h – number of past frames in the input window (default 200).
        Corresponds to N_h = T_h / Δt (e.g. 20 s / 0.1 s = 200).
    prediction_horizon : int
        H – number of future steps to predict (default 50).
        Corresponds to N_p = H_sec / Δt (e.g. 5 s / 0.1 s = 50).
    hidden_units : int
        LSTM units per direction (default 128).
    n_lstm_layers : int
        Number of stacked Bi-LSTM layers (default 2).
    dropout_rate : float
        Dropout for LSTM layers (default 0.1).
    l2_reg : float
        L2 regularization coefficient (default 1e-4).
    lambda_snr : float
        Weight for SNR prediction loss (default 1.0).
    lambda_att : float
        Weight for attenuation prediction loss (default 0.0).
    lambda_fm : float
        Weight for fade-margin loss (default 0.0).
    learning_rate : float
        Adam optimizer learning rate (default 1e-3).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for training.

    Input Shapes
    -------------
    pilot_blocks : (batch, history_len, B, N_max, C_pilot)
    pilot_masks  : (batch, history_len, B, N_max)
    modcod_features : (batch, history_len, 2)  →  [M_k, R_k]

    Output Shape
    ------------
    snr_predictions : (batch, prediction_horizon)
    """

    # ---- Inputs ----------------------------------------------------------
    pilot_blocks_input = layers.Input(
        shape=(history_len, num_blocks, n_max, c_pilot),
        name="pilot_blocks",
    )
    pilot_masks_input = layers.Input(
        shape=(history_len, num_blocks, n_max),
        name="pilot_masks",
    )
    modcod_input = layers.Input(
        shape=(history_len, 2),
        name="modcod_features",
    )

    # ---- Stage 1: CNN Feature Extraction (applied per time step) ---------
    aggregator = PilotBlockAggregator(
        num_blocks=num_blocks,
        n_max=n_max,
        d_f=d_f,
        d_s=d_s,
        kernel_size=cnn_kernel_size,
        l2_reg=l2_reg,
        name="stage1_aggregator",
    )

    # Apply Stage 1 across all time steps (batch+time merge strategy)
    stage1_td = Stage1TimeDistributed(
        aggregator=aggregator,
        history_len=history_len,
        name="stage1_time_distributed",
    )
    channel_features = stage1_td(
        [pilot_blocks_input, pilot_masks_input]
    )  # (batch, history_len, d_s)

    # ---- Combine channel features with MODCOD context --------------------
    # x_k = [s_k, M_k, R_k]  →  d_in = d_s + 2
    combined = layers.Concatenate(axis=-1, name="combine_features")(
        [channel_features, modcod_input]
    )  # (batch, history_len, d_s + 2)

    # ---- Stage 2: Bi-LSTM Temporal Prediction ----------------------------
    predictor = BiLSTMPredictor(
        hidden_units=hidden_units,
        n_layers=n_lstm_layers,
        prediction_horizon=prediction_horizon,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name="stage2_predictor",
    )

    snr_predictions = predictor(combined)  # (batch, H)

    # ---- Assemble Model --------------------------------------------------
    model = Model(
        inputs=[pilot_blocks_input, pilot_masks_input, modcod_input],
        outputs=snr_predictions,
        name="TwoStage_CNN_BiLSTM_ACM",
    )

    # ---- Compile ---------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


# ---------------------------------------------------------------------------
# Lightweight Stage-2-only model (when Stage 1 features are pre-computed)
# ---------------------------------------------------------------------------
def build_stage2_only_model(
    d_in: int = 7,
    history_len: int = 200,
    prediction_horizon: int = 50,
    hidden_units: int = 128,
    n_lstm_layers: int = 2,
    dropout_rate: float = 0.1,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> Model:
    """Build a standalone Stage-2 Bi-LSTM model.

    Use this when channel-state features have already been extracted
    (e.g. from a pre-trained Stage 1 or from analytical estimation).

    Parameters
    ----------
    d_in : int
        Input feature dimension per time step (d_s + 2 = 7 by default).
    history_len : int
        Number of past time steps in the input window (default 200).
    prediction_horizon : int
        Number of future steps to predict (default 50).
    hidden_units : int
        LSTM units per direction (default 128).
    n_lstm_layers : int
        Number of Bi-LSTM layers (default 2).
    dropout_rate : float
        Dropout rate (default 0.1).
    l2_reg : float
        L2 regularization (default 1e-4).
    learning_rate : float
        Adam learning rate (default 1e-3).

    Returns
    -------
    model : tf.keras.Model
        Compiled model.

    Input Shape
    -----------
    (batch, history_len, d_in)

    Output Shape
    ------------
    (batch, prediction_horizon)
    """
    inp = layers.Input(shape=(history_len, d_in), name="input_features")

    predictor = BiLSTMPredictor(
        hidden_units=hidden_units,
        n_layers=n_lstm_layers,
        prediction_horizon=prediction_horizon,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name="stage2_predictor",
    )

    out = predictor(inp)

    model = Model(inputs=inp, outputs=out, name="Stage2_BiLSTM_Predictor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


# ---------------------------------------------------------------------------
# ETSI MODCOD Lookup Table
# ---------------------------------------------------------------------------
# Required Es/N0 for 10^-6 PER, AWGN (dB) – from the diagram
ETSI_MODCOD_TABLE = [
    # (modulation, code_rate_str, code_rate_float, required_esn0_db)
    ("QPSK",   "1/4",  0.25,   -2.0),
    ("QPSK",   "1/2",  0.50,    0.0),
    ("QPSK",   "3/4",  0.75,    1.8),
    ("QPSK",   "2/3",  0.667,   4.0),
    ("8PSK",   "3/4",  0.75,    5.0),
    ("8PSK",   "5/6",  0.833,   6.0),
    ("8PSK",   "2/3",  0.667,   7.5),
    ("16APSK", "3/4",  0.75,    8.8),
    ("16APSK", "5/6",  0.833,   9.8),
    ("16APSK", "3/4",  0.75,   12.0),
    ("32APSK", "3/4",  0.75,   13.5),
    ("32APSK", "5/6",  0.833,  15.0),
    ("32APSK", "8/9",  0.889,  15.0),
    ("32APSK", "9/10", 0.90,   13.0),
]


def select_modcod(predicted_snr_db: float, margin_db: float = 0.0) -> dict:
    """Select the highest-throughput MODCOD whose required SNR ≤ predicted SNR − margin.

    This implements the ACM decision logic from the diagram:
        For each future step, select the highest MODCOD whose
        required SNR ≤ (Predicted SNR − Margin).

    Parameters
    ----------
    predicted_snr_db : float
        Predicted SNR in dB for the target future time step.
    margin_db : float
        Implementation margin M (dB) for safety (default 0.0).

    Returns
    -------
    dict with keys: modulation, code_rate, code_rate_float, required_esn0_db
        The selected MODCOD entry, or the most robust one if none qualifies.
    """
    available_snr = predicted_snr_db - margin_db

    # Sort by required Es/N0 descending to find the highest feasible MODCOD
    sorted_table = sorted(ETSI_MODCOD_TABLE, key=lambda x: x[3], reverse=True)

    for mod, cr_str, cr_float, req_esn0 in sorted_table:
        if req_esn0 <= available_snr:
            return {
                "modulation": mod,
                "code_rate": cr_str,
                "code_rate_float": cr_float,
                "required_esn0_db": req_esn0,
            }

    # Fall back to the most robust MODCOD (lowest required SNR)
    fallback = min(ETSI_MODCOD_TABLE, key=lambda x: x[3])
    return {
        "modulation": fallback[0],
        "code_rate": fallback[1],
        "code_rate_float": fallback[2],
        "required_esn0_db": fallback[3],
    }


# ---------------------------------------------------------------------------
# Convenience: print model summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  Building Full Two-Stage CNN + Bi-LSTM Model")
    print("=" * 70)

    full_model = build_two_stage_model(
        num_blocks=22,
        n_max=36,
        c_pilot=2,
        d_f=64,
        d_s=5,
        history_len=200,
        prediction_horizon=50,
        hidden_units=128,
        n_lstm_layers=2,
    )
    full_model.summary(line_length=100, expand_nested=True)

    print("\n" + "=" * 70)
    print("  Building Stage-2 Only Bi-LSTM Model")
    print("=" * 70)

    s2_model = build_stage2_only_model(
        d_in=7,
        history_len=200,
        prediction_horizon=50,
        hidden_units=128,
        n_lstm_layers=2,
    )
    s2_model.summary(line_length=100)

    print("\n" + "=" * 70)
    print("  MODCOD Selection Example")
    print("=" * 70)
    for snr in [0.0, 5.0, 10.0, 14.0]:
        result = select_modcod(snr, margin_db=1.0)
        print(f"  Predicted SNR = {snr:5.1f} dB  ->  {result['modulation']:>7s} "
              f"{result['code_rate']:>4s}  (req {result['required_esn0_db']:.1f} dB)")
