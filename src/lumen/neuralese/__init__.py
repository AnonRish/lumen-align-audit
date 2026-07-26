from .tensor import Tensor, Adam, embedding_lookup, cross_entropy
from .toy_transformer import (
    ToyTransformer, ToyTransformerConfig, train_toy_transformer, make_batch,
    encode_example, true_answer, evaluate, TOKEN_NAMES, SECRET_OFFSET_DEFAULT,
)
from .logit_lens import run_logit_lens, TunedLens, LogitLensResult, softmax_np, kl_divergence
from .patching import run_activation_patching, PatchResult
from .sae import SparseAutoencoder, SAEConfig, train_sae, feature_mode_correlation, SAETrainResult
from .recurrent_reasoner import (
    NeuraleseRecurrentReasoner, RecurrentReasonerConfig, train_recurrent_reasoner,
    evaluate_recurrent, make_recurrent_batch, apply_op, encode_op, RecurrentTrainResult,
)
from .recurrent_decoding import (
    run_stepwise_logit_lens, run_stepwise_patching, StepwiseLogitLensResult, StepwisePatchingResult,
)

__all__ = [
    "Tensor", "Adam", "embedding_lookup", "cross_entropy",
    "ToyTransformer", "ToyTransformerConfig", "train_toy_transformer", "make_batch",
    "encode_example", "true_answer", "evaluate", "TOKEN_NAMES", "SECRET_OFFSET_DEFAULT",
    "run_logit_lens", "TunedLens", "LogitLensResult", "softmax_np", "kl_divergence",
    "run_activation_patching", "PatchResult",
    "SparseAutoencoder", "SAEConfig", "train_sae", "feature_mode_correlation", "SAETrainResult",
    "NeuraleseRecurrentReasoner", "RecurrentReasonerConfig", "train_recurrent_reasoner",
    "evaluate_recurrent", "make_recurrent_batch", "apply_op", "encode_op", "RecurrentTrainResult",
    "run_stepwise_logit_lens", "run_stepwise_patching", "StepwiseLogitLensResult", "StepwisePatchingResult",
]
