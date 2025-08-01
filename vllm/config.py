# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import enum
import hashlib
import inspect
import json
import textwrap
import uuid
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import (MISSING, Field, asdict, field, fields, is_dataclass,
                         replace)
from functools import cached_property
from importlib.util import find_spec
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Literal, Optional,
                    Protocol, TypeVar, Union, cast, get_args)

import regex as re
import torch
from pydantic import (ConfigDict, SkipValidation, TypeAdapter, field_validator,
                      model_validator)
from pydantic.dataclasses import dataclass
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self, assert_never, runtime_checkable

import vllm.envs as envs
from vllm import version
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    try_get_generation_config, try_get_safetensors_metadata,
    try_get_tokenizer_config, uses_mrope)
from vllm.transformers_utils.s3_utils import S3Model
from vllm.transformers_utils.utils import is_s3, maybe_model_redirect
# yapf conflicts with isort for this block
# yapf: disable
from vllm.utils import (DEFAULT_MAX_NUM_BATCHED_TOKENS,
                        MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                        POOLING_MODEL_MAX_NUM_BATCHED_TOKENS, GiB_bytes,
                        LayerBlockType, LazyLoader, common_broadcastable_dtype,
                        cuda_device_count_stateless, get_cpu_memory,
                        get_open_port, is_torch_equal_or_newer, random_uuid,
                        resolve_obj_by_qualname)

# yapf: enable

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from ray.util.placement_group import PlacementGroup
    from transformers.configuration_utils import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    import vllm.model_executor.models as me_models
    from vllm.executor.executor_base import ExecutorBase
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    from vllm.model_executor.model_loader import LoadFormats
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

    ConfigType = type[DataclassInstance]
    HfOverrides = Union[dict, Callable[[type], type]]
else:
    DataclassInstance = Any
    PlacementGroup = Any
    PretrainedConfig = Any
    ExecutorBase = Any
    QuantizationConfig = Any
    QuantizationMethods = Any
    BaseModelLoader = Any
    LoadFormats = Any
    TensorizerConfig = Any
    ConfigType = type
    HfOverrides = Union[dict[str, Any], Callable[[type], type]]

    me_quant = LazyLoader("model_executor", globals(),
                          "vllm.model_executor.layers.quantization")
    me_models = LazyLoader("model_executor", globals(),
                           "vllm.model_executor.models")

logger = init_logger(__name__)
DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)
ConfigT = TypeVar("ConfigT", bound=ConfigType)

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward", "transcription", "draft"]

_ResolvedTask = Literal["generate", "transcription", "encode", "embed",
                        "classify", "reward", "draft"]

RunnerOption = Literal["auto", "generate", "pooling", "draft"]

RunnerType = Literal["generate", "pooling", "draft"]

ConvertOption = Literal["auto", "none", "embed", "classify", "reward"]

ConvertType = Literal["none", "embed", "classify", "reward"]

_RUNNER_TASKS: dict[RunnerType, list[TaskOption]] = {
    "generate": ["generate", "transcription"],
    "pooling": ["embedding", "embed", "classify", "score", "reward"],
    "draft": ["draft"],
}

_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed", "classify", "reward"],
    "draft": [],
}

# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "reward")),
    ("RewardModel", ("pooling", "reward")),
    # Let other `*Model`s take priority
    ("Model", ("pooling", "embed")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
    architecture: str,
    *,
    runner_type: Optional[RunnerType] = None,
    convert_type: Optional[ConvertType] = None,
) -> Optional[tuple[str, tuple[RunnerType, ConvertType]]]:
    for suffix, (default_runner_type,
                 default_convert_type) in iter_architecture_defaults():
        if ((runner_type is None or runner_type == default_runner_type) and
            (convert_type is None or convert_type == default_convert_type)
                and architecture.endswith(suffix)):
            return suffix, (default_runner_type, default_convert_type)

    return None


@runtime_checkable
class SupportsHash(Protocol):

    def compute_hash(self) -> str:
        ...


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> dict[str, str]:
        ...


class ModelImpl(str, enum.Enum):
    AUTO = "auto"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    def pairwise(iterable):
        """
        Manually implement https://docs.python.org/3/library/itertools.html#itertools.pairwise

        Can be removed when Python 3.9 support is dropped.
        """
        iterator = iter(iterable)
        a = next(iterator, None)

        for b in iterator:
            yield a, b
            a = b

    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (not isinstance(a, (ast.Assign, ast.AnnAssign))
                or not isinstance(b, ast.Expr)
                or not isinstance(b.value, ast.Constant)
                or not isinstance(b.value.value, str)):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the default value provided
    by `get_kwargs` will be the result parsing a JSON string as the kwargs
    (i.e. `ConfigT(**json.loads(cli_arg))`). However, if a particular `ConfigT`
    requires custom construction from CLI (i.e. `CompilationConfig`), it can
    have a `from_cli` method, which will be called instead.

    Config validation is performed by the tools/validate_config.py
    script, which is invoked during the pre-commit checks.
    """
    return cls


def get_field(cls: ConfigType, name: str) -> Field:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    cls_fields = {f.name: f for f in fields(cls)}
    if name not in cls_fields:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
    named_field: Field = cls_fields[name]
    if (default_factory := named_field.default_factory) is not MISSING:
        return field(default_factory=default_factory)
    if (default := named_field.default) is not MISSING:
        return field(default=default)
    raise ValueError(
        f"{cls.__name__}.{name} must have a default value or default factory.")


def is_init_field(cls: ConfigType, name: str) -> bool:
    return next(f for f in fields(cls) if f.name == name).init


TokenizerMode = Literal["auto", "slow", "mistral", "custom"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
LogprobsMode = Literal["raw_logprobs", "raw_logits", "processed_logprobs",
                       "processed_logits"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""

    model: str = "Qwen/Qwen3-0.6B"
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    runner: RunnerOption = "auto"
    """The type of model runner to use. Each vLLM instance only supports one
    model runner, even if the same model can be used for multiple types."""
    convert: ConvertOption = "auto"
    """Convert the model using adapters defined in
    [vllm.model_executor.models.adapters][]. The most common use case is to
    adapt a text generation model to be used for pooling tasks."""
    task: Optional[TaskOption] = None
    """[DEPRECATED] The task to use the model for. If the model supports more
    than one model runner, this is used to select which model runner to run.

    Note that the model may support other tasks using the same model runner.
    """
    tokenizer: SkipValidation[str] = None  # type: ignore
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode = "auto"
    """Tokenizer mode:\n
    - "auto" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "custom" will use --tokenizer to select the preregistered tokenizer."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: Union[ModelDType, torch.dtype] = "auto"
    """Data type for model weights and activations:\n
    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
    precision for BF16 models.\n
    - "half" for FP16. Recommended for AWQ quantization.\n
    - "float16" is the same as "half".\n
    - "bfloat16" for a balance between precision and range.\n
    - "float" is shorthand for FP32 precision.\n
    - "float32" for FP32 precision."""
    seed: Optional[int] = None
    """Random seed for reproducibility. Initialized to None in V0, but
    initialized to 0 in V1."""
    hf_config_path: Optional[str] = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    revision: Optional[str] = None
    """The specific model version to use. It can be a branch name, a tag name,
    or a commit id. If unspecified, will use the default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the model code on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    rope_scaling: dict[str, Any] = field(default_factory=dict)
    """RoPE scaling configuration. For example,
    `{"rope_type":"dynamic","factor":2.0}`."""
    rope_theta: Optional[float] = None
    """RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE
    theta improves the performance of the scaled model."""
    tokenizer_revision: Optional[str] = None
    """The specific revision to use for the tokenizer on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    max_model_len: SkipValidation[int] = None  # type: ignore
    """Model context length (prompt and output). If unspecified, will be
    automatically derived from the model config.

    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format. Examples:\n
    - 1k -> 1000\n
    - 1K -> 1024\n
    - 25.6k -> 25,600"""
    spec_target_max_model_len: Optional[int] = None
    """Specify the maximum length for spec decoding draft models."""
    quantization: SkipValidation[Optional[QuantizationMethods]] = None
    """Method used to quantize the weights. If `None`, we first check the
    `quantization_config` attribute in the model config file. If that is
    `None`, we assume the model weights are not quantized and use `dtype` to
    determine the data type of the weights."""
    enforce_eager: bool = False
    """Whether to always use eager-mode PyTorch. If True, we will disable CUDA
    graph and always execute the model in eager mode. If False, we will use
    CUDA graph and eager execution in hybrid for maximal performance and
    flexibility."""
    max_seq_len_to_capture: int = 8192
    """Maximum sequence len covered by CUDA graphs. When a sequence has context
    length larger than this, we fall back to eager mode. Additionally for
    encoder-decoder models, if the sequence length of the encoder input is
    larger than this, we fall back to the eager mode."""
    max_logprobs: int = 20
    """Maximum number of log probabilities to return when `logprobs` is
    specified in `SamplingParams`. The default value comes the default for the
    OpenAI Chat Completions API."""
    logprobs_mode: LogprobsMode = "raw_logprobs"
    """Indicates the content returned in the logprobs and prompt_logprobs.
    Supported mode:
    1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    Raw means the values before applying logit processors, like bad words.
    Processed means the values after applying such processors.
    """
    disable_sliding_window: bool = False
    """Whether to disable sliding window. If True, we will disable the sliding
    window functionality of the model, capping to sliding window size. If the
    model does not support sliding window, this argument is ignored."""
    disable_cascade_attn: bool = False
    """Disable cascade attention for V1. While cascade attention does not
    change the mathematical correctness, disabling it could be useful for
    preventing potential numerical issues. Note that even if this is set to
    False, cascade attention will be only used when the heuristic tells that
    it's beneficial."""
    skip_tokenizer_init: bool = False
    """Skip initialization of tokenizer and detokenizer. Expects valid
    `prompt_token_ids` and `None` for prompt from the input. The generated
    output will contain token ids."""
    enable_prompt_embeds: bool = False
    """If `True`, enables passing text embeddings as inputs via the
    `prompt_embeds` key. Note that enabling this will double the time required
    for graph compilation."""
    served_model_name: Optional[Union[str, list[str]]] = None
    """The model name(s) used in the API. If multiple names are provided, the
    server will respond to any of the provided names. The model name in the
    model field of a response will be the first name in this list. If not
    specified, the model name will be the same as the `--model` argument. Noted
    that this name(s) will also be used in `model_name` tag content of
    prometheus metrics, if multiple names provided, metrics tag will take the
    first one."""
    limit_mm_per_prompt: dict[str, int] = field(default_factory=dict)
    """Maximum number of data items per modality per prompt. Only applicable
    for multimodal models."""
    interleave_mm_strings: bool = False
    """Enable fully interleaved support for multimodal prompts, while using
    --chat-template-content-format=string. Defaults to False."""
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Additional args passed to process media inputs, keyed by modalities.
    For example, to set num_frames for video, set
    `--media-io-kwargs '{"video": {"num_frames": 40} }'` """
    use_async_output_proc: bool = True
    """Whether to use async output processor."""
    config_format: Union[str, ConfigFormat] = ConfigFormat.AUTO.value
    """The format of the model config to load:\n
    - "auto" will try to load the config in hf format if available else it
    will try to load in mistral format.\n
    - "hf" will load the config in hf format.\n
    - "mistral" will load the config in mistral format."""
    hf_token: Optional[Union[bool, str]] = None
    """The token to use as HTTP bearer authorization for remote files . If
    `True`, will use the token generated when running `huggingface-cli login`
    (stored in `~/.huggingface`)."""
    hf_overrides: HfOverrides = field(default_factory=dict)
    """If a dictionary, contains arguments to be forwarded to the Hugging Face
    config. If a callable, it is called to update the HuggingFace config."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Arguments to be forwarded to the model's processor for multi-modal data,
    e.g., image processor. Overrides for the multi-modal processor obtained
    from `AutoProcessor.from_pretrained`. The available overrides depend on the
    model that is being run. For example, for Phi-3-Vision: `{"num_crops": 4}`.
    """
    disable_mm_preprocessor_cache: bool = False
    """If `True`, disable caching of the multi-modal preprocessor/mapper (not
    recommended)."""
    override_neuron_config: dict[str, Any] = field(default_factory=dict)
    """Initialize non-default neuron config or override default neuron config
    that are specific to Neuron devices, this argument will be used to
    configure the neuron config that can not be gathered from the vllm
    arguments. e.g. `{"cast_logits_dtype": "bfloat16"}`."""
    pooler_config: Optional["PoolerConfig"] = field(init=False)
    """Pooler config which controls the behaviour of output pooling in pooling
    models."""
    override_pooler_config: Optional[Union[dict, "PoolerConfig"]] = None
    """Initialize non-default pooling config or override default pooling config
    for the pooling model. e.g. `{"pooling_type": "mean", "normalize": false}`.
    """
    logits_processor_pattern: Optional[str] = None
    """Optional regex pattern specifying valid logits processor qualified names
    that can be passed with the `logits_processors` extra completion argument.
    Defaults to `None`, which allows no processors."""
    generation_config: str = "auto"
    """The folder path to the generation config. Defaults to `"auto"`, the
    generation config will be loaded from model path. If set to `"vllm"`, no
    generation config is loaded, vLLM defaults will be used. If set to a folder
    path, the generation config will be loaded from the specified folder path.
    If `max_new_tokens` is specified in generation config, then it sets a
    server-wide limit on the number of output tokens for all requests."""
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    """Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If
    used with `--generation-config auto`, the override parameters will be
    merged with the default config from the model. If used with
    `--generation-config vllm`, only the override parameters are used."""
    enable_sleep_mode: bool = False
    """Enable sleep mode for the engine (only cuda platform is supported)."""
    model_impl: Union[str, ModelImpl] = ModelImpl.AUTO.value
    """Which implementation of the model to use:\n
    - "auto" will try to use the vLLM implementation, if it exists, and fall
    back to the Transformers implementation if no vLLM implementation is
    available.\n
    - "vllm" will use the vLLM model implementation.\n
    - "transformers" will use the Transformers model implementation."""
    override_attention_dtype: Optional[str] = None
    """Override dtype for attention"""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.model)
        factors.append(self.dtype)
        factors.append(self.quantization)
        factors.append(self.revision)
        factors.append(self.code_revision)
        factors.append(self.max_model_len)
        factors.append(self.max_logprobs)
        factors.append(self.disable_sliding_window)
        factors.append(self.trust_remote_code)
        factors.append(self.generation_config)
        factors.append(self.model_impl)
        factors.append(self.override_generation_config)
        factors.append(self.rope_scaling)
        factors.append(self.rope_theta)
        # hf_config can control how the model looks!
        factors.append(self.hf_config.to_json_string())
        str_factors = str(factors)
        assert_hashable(str_factors)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        # Set the default seed to 0 in V1.
        # NOTE(woosuk): In V0, we set the default seed to None because the
        # driver worker shares the same process as the user process, and thus
        # setting a seed affects the user process as well.
        # In V1, we use separate processes for workers (unless
        # VLLM_ENABLE_V1_MULTIPROCESSING=0), so setting a seed here
        # doesn't affect the user process. However, without a consistent seed,
        # different tensor parallel workers would sample different tokens,
        # leading to inconsistent results.
        if envs.VLLM_USE_V1 and self.seed is None:
            self.seed = 0
            if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
                logger.warning(
                    "The global random seed is set to %d. Since "
                    "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                    "affect the random state of the Python process that "
                    "launched vLLM.", self.seed)

        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(self.model,
                                                       self.served_model_name)
        self.model = maybe_model_redirect(self.model)
        # The tokenizer is consistent with the model by default.
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        if callable(self.hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = self.hf_overrides
        else:
            hf_overrides_kw = self.hf_overrides
            hf_overrides_fn = None

        if self.rope_scaling:
            hf_override: dict[str, Any] = {"rope_scaling": self.rope_scaling}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = (
                "`--rope-scaling` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if self.rope_theta is not None:
            hf_override = {"rope_theta": self.rope_theta}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = (
                "`--rope-theta` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        self.maybe_pull_model_tokenizer_for_s3(self.model, self.tokenizer)

        if (backend := envs.VLLM_ATTENTION_BACKEND
            ) and backend == "FLASHINFER" and find_spec("flashinfer") is None:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer "
                "module was not found. See "
                "https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile "  # noqa: E501
                "for instructions on how to install it.")

        from vllm.platforms import current_platform

        if (self.override_attention_dtype is not None
                and not current_platform.is_rocm()):
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2)

        if (self.enable_sleep_mode
                and not current_platform.is_sleep_mode_available()):
            raise ValueError(
                "Sleep mode is not supported on current platform.")

        if isinstance(self.config_format, str):
            self.config_format = ConfigFormat(self.config_format)

        hf_config = get_config(self.hf_config_path or self.model,
                               self.trust_remote_code,
                               self.revision,
                               self.code_revision,
                               self.config_format,
                               hf_overrides_kw=hf_overrides_kw,
                               hf_overrides_fn=hf_overrides_fn)
        self.hf_config = hf_config

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(self.hf_text_config,
                                            "attention_chunk_size", None)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision)

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(
            architectures, self)
        is_pooling_model = registry.is_pooling_model(architectures, self)

        def _task_to_convert(task: TaskOption) -> ConvertType:
            if task == "embedding" or task == "embed":
                return "embed"
            if task == "classify":
                return "classify"
            if task == "reward":
                return "reward"
            if task == "score":
                new_task = self._get_default_pooling_task(architectures)
                return "classify" if new_task == "classify" else "embed"

            return "none"

        if self.task is not None:
            runner: RunnerOption = "auto"
            convert: ConvertOption = "auto"
            msg_prefix = ("The 'task' option has been deprecated and will be "
                          "removed in v0.13.0 or v1.0, whichever comes first.")
            msg_hint = "Please remove this option."

            is_generative_task = self.task in _RUNNER_TASKS["generate"]
            is_pooling_task = self.task in _RUNNER_TASKS["pooling"]

            if is_generative_model and is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = ("Please replace this option with `--runner "
                                "generate` to continue using this model "
                                "as a generative model.")
                elif is_pooling_task:
                    runner = "pooling"
                    convert = "auto"
                    msg_hint = ("Please replace this option with `--runner "
                                "pooling` to continue using this model "
                                "as a pooling model.")
                else:  # task == "auto"
                    pass
            elif is_generative_model or is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = "Please remove this option"
                elif is_pooling_task:
                    runner = "pooling"
                    convert = _task_to_convert(self.task)
                    msg_hint = ("Please replace this option with `--convert "
                                f"{convert}` to continue using this model "
                                "as a pooling model.")
                else:  # task == "auto"
                    pass
            else:
                raise AssertionError("The model should be a generative or "
                                     "pooling model when task is set to "
                                     f"{self.task!r}.")

            self.runner = runner
            self.convert = convert

            msg = f"{msg_prefix} {msg_hint}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        self.runner_type = self._get_runner_type(architectures, self.runner)
        self.convert_type = self._get_convert_type(architectures,
                                                   self.runner_type,
                                                   self.convert)

        if self.runner_type == "generate" and not is_generative_model:
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError(
                    "This model does not support `--runner generate`.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model.")

        self.supported_tasks = self._get_supported_tasks(
            architectures, self.runner_type, self.convert_type)

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        self.pooler_config = self._init_pooler_config()

        self.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        # Workaround for Gemma 2 which uses interleaved sliding window
        # attention, but it's not specified in its config. TODO: remove this
        # when Gemma 2 is fixed in Transformers.
        if self.hf_text_config.model_type == "gemma2":
            self.hf_text_config.sliding_window_pattern = 2

        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        sliding_window_pattern = getattr(self.hf_text_config,
                                         "sliding_window_pattern", None)
        has_interleaved_attention = sliding_window_pattern is not None or (
            isinstance(sliding_window, list))

        if not self.disable_sliding_window and has_interleaved_attention:
            if (backend :=
                    envs.VLLM_ATTENTION_BACKEND) in ("XFORMERS", "FLASHINFER"):
                sliding_window_len_min = get_min_sliding_window(
                    self.hf_text_config.sliding_window)

                logger.warning_once(
                    "%s has interleaved attention, which is currently not supported by the %s backend. Disabling sliding window and capping the max length to the sliding window size (%d).",  # noqa: E501
                    self.hf_text_config.model_type,
                    backend,
                    sliding_window_len_min,
                )
                self.disable_sliding_window = True
            else:
                # for a model with interleaved attention,
                # the scheduler and the model treat it as full attention
                # (i.e., not dropping any tokens outside the window).
                # only the attention layer itself is aware of the sliding
                # window, and use the window size to compute the attention.
                self.hf_text_config.interleaved_sliding_window = sliding_window

                if hasattr(self.hf_text_config, "sliding_window"):
                    delattr(self.hf_text_config, "sliding_window")

                sliding_window = None

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
        self.multimodal_config = self._init_multimodal_config()

        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        if (not current_platform.is_neuron() and self.override_neuron_config):
            raise ValueError(
                "`override_neuron_config` is only supported on Neuron.")

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization_before(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def validate_model_config_after(self: "ModelConfig") -> "ModelConfig":
        if not isinstance(self.tokenizer, str):
            raise ValueError("tokenizer must be a string after __post_init__.")
        if not isinstance(self.max_model_len, int):
            raise ValueError(
                "max_model_len must be an integer after __post_init__.")
        return self

    def _get_transformers_backend_cls(self) -> str:
        """Determine which Transformers backend class will be used if
        `model_impl` is set to `transformers` or `auto`."""
        if self.hf_config != self.hf_text_config:
            # If 'hf_text_config' is the same as 'hf_config'. If not, it is
            # probably a composite config, i.e. multimodal
            return "TransformersForMultimodalLM"
        else:
            return "TransformersForCausalLM"

    @property
    def registry(self):
        return me_models.ModelRegistry

    @property
    def architectures(self) -> list[str]:
        return getattr(self.hf_config, "architectures", [])

    @property
    def architecture(self) -> str:
        """The architecture vllm actually used."""
        return self._architecture

    def maybe_pull_model_tokenizer_for_s3(self, model: str,
                                          tokenizer: str) -> None:
        """Pull model/tokenizer from S3 to temporary directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """
        if not (is_s3(model) or is_s3(tokenizer)):
            return

        if is_s3(model):
            s3_model = S3Model()
            s3_model.pull_files(model,
                                allow_pattern=["*.model", "*.py", "*.json"])
            self.model_weights = model
            self.model = s3_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                s3_model.pull_files(model,
                                    ignore_pattern=[
                                        "*.pt", "*.safetensors", "*.bin",
                                        "*.tensors"
                                    ])
                self.tokenizer = s3_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_s3(tokenizer):
            s3_tokenizer = S3Model()
            s3_tokenizer.pull_files(
                model,
                ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors"])
            self.tokenizer = s3_tokenizer.dir

    def _init_multimodal_config(self) -> Optional["MultiModalConfig"]:
        if self.registry.is_multimodal_model(self.architectures, self):
            return MultiModalConfig(
                limit_per_prompt=self.limit_mm_per_prompt,
                media_io_kwargs=self.media_io_kwargs,
                mm_processor_kwargs=self.mm_processor_kwargs,
                disable_mm_preprocessor_cache=self.
                disable_mm_preprocessor_cache,
                interleave_mm_strings=self.interleave_mm_strings)

        if self.limit_mm_per_prompt:
            raise ValueError("`limit_mm_per_prompt` is only supported for "
                             "multimodal models.")
        if self.mm_processor_kwargs:
            raise ValueError("`mm_processor_kwargs` is only supported for "
                             "multimodal models.")
        if self.disable_mm_preprocessor_cache:
            raise ValueError("`disable_mm_preprocessor_cache` is only "
                             "supported for multimodal models.")
        if self.interleave_mm_strings:
            raise ValueError("`interleave_mm_strings` is only "
                             "supported for multimodal models.")

        return None

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(
            self.model, self.revision)

    def _init_pooler_config(self) -> Optional["PoolerConfig"]:
        if self.runner_type == "pooling":
            if isinstance(self.override_pooler_config, dict):
                self.override_pooler_config = PoolerConfig(
                    **self.override_pooler_config)

            pooler_config = self.override_pooler_config or PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(pooler_config, k) is None:
                        setattr(pooler_config, k, v)

            if self.is_matryoshka:
                if pooler_config.normalize is None:
                    pooler_config.normalize = True
                elif not pooler_config.normalize:
                    raise ValueError(
                        "`normalize` must be enabled (set to True) "
                        "for models that are compatible with "
                        "Matryoshka Representation.")

            return pooler_config

        return None

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = cast(TokenizerMode, self.tokenizer_mode.lower())
        if tokenizer_mode not in get_args(TokenizerMode):
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                f"one of {get_args(TokenizerMode)}.")
        self.tokenizer_mode = tokenizer_mode

    def _get_default_runner_type(
        self,
        architectures: list[str],
    ) -> RunnerType:
        registry = self.registry

        # Some Sentence Transformers models use *ForCausalLM archs
        if get_pooling_config(self.model, self.revision):
            return "pooling"

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if registry.is_pooling_model(architectures, self):
                    return "pooling"
                if registry.is_text_generation_model(architectures, self):
                    return "generate"

            match = try_match_architecture_defaults(arch)
            if match:
                _, (runner_type, _) = match
                return runner_type

        return "generate"

    def _get_runner_type(
        self,
        architectures: list[str],
        runner: RunnerOption,
    ) -> RunnerType:
        if runner != "auto":
            return runner

        runner_type = self._get_default_runner_type(architectures)

        logger.info(
            "Resolved `--runner auto` to `--runner %s`. "
            "Pass the value explicitly to silence this message.", runner_type)

        return runner_type

    def _get_default_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
    ) -> ConvertType:
        registry = self.registry

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if (runner_type == "generate"
                        and registry.is_text_generation_model(
                            architectures, self)):
                    return "none"
                if (runner_type == "pooling"
                        and registry.is_pooling_model(architectures, self)):
                    return "none"

            match = try_match_architecture_defaults(arch,
                                                    runner_type=runner_type)
            if match:
                _, (_, convert_type) = match
                return convert_type

        # This is to handle Sentence Transformers models that use *ForCausalLM
        # and also multi-modal pooling models which are not defined as
        # Sentence Transformers models
        if runner_type == "pooling":
            return "embed"

        return "none"

    def _get_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert: ConvertOption,
    ) -> ConvertType:
        if convert != "auto":
            return convert

        convert_type = self._get_default_convert_type(architectures,
                                                      runner_type)

        logger.info(
            "Resolved `--convert auto` to `--convert %s`. "
            "Pass the value explicitly to silence this message.", convert_type)

        return convert_type

    def _get_supported_generation_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        if registry.is_transcription_only_model(architectures, self):
            return ["transcription"]

        # TODO: Use get_supported_generation_tasks once V0 is removed
        supported_tasks = list[_ResolvedTask]()
        if (registry.is_text_generation_model(architectures, self)
                or convert_type in _RUNNER_CONVERTS["generate"]):
            supported_tasks.append("generate")

        if registry.is_transcription_model(architectures, self):
            supported_tasks.append("transcription")

        return supported_tasks

    def _get_default_pooling_task(
        self,
        architectures: list[str],
    ) -> Literal["embed", "classify", "reward"]:
        if self.registry.is_cross_encoder_model(architectures, self):
            return "classify"

        for arch in architectures:
            match = try_match_architecture_defaults(arch,
                                                    runner_type="pooling")
            if match:
                _, (_, convert_type) = match
                assert convert_type != "none"
                return convert_type

        return "embed"

    def _get_supported_pooling_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        # TODO: Use get_supported_pooling_tasks once V0 is removed
        supported_tasks = list[_ResolvedTask]()
        if (registry.is_pooling_model(architectures, self)
                or convert_type in _RUNNER_CONVERTS["pooling"]):
            supported_tasks.append("encode")

            extra_task = (self._get_default_pooling_task(architectures)
                          if convert_type == "none" else convert_type)
            supported_tasks.append(extra_task)

        return supported_tasks

    def _get_supported_tasks(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        if runner_type == "generate":
            return self._get_supported_generation_tasks(
                architectures, convert_type)
        if runner_type == "pooling":
            return self._get_supported_pooling_tasks(architectures,
                                                     convert_type)
        if runner_type == "draft":
            return ["draft"]

        assert_never(runner_type)

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed-tensors", "experts_int8",
            "quark", "modelopt_fp4", "bitblas", "gptq_bitblas", "inc"
        ]
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods,
                                     self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace("compressed_tensors",
                                                "compressed-tensors")

            quant_cfg["quant_method"] = quant_method

            # Quantization methods which are overrides (i.e. they have a
            # `override_quantization_method` method) must be checked in order
            # of preference (this is particularly important for GPTQ).
            overrides = [
                "marlin",
                "bitblas",
                "gptq_marlin_24",
                "gptq_marlin",
                "gptq_bitblas",
                "awq_marlin",
                "ipex",
                "moe_wna16",
                "modelopt",
                "modelopt_fp4",
            ]
            quantization_methods = [
                q for q in supported_quantization if q not in overrides
            ]
            # Any custom overrides will be in quantization_methods so we place
            # them at the start of the list so custom overrides have preference
            # over the built in ones.
            quantization_methods = quantization_methods + overrides

            # Detect which checkpoint is it
            for name in quantization_methods:
                method = me_quant.get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (name in get_args(me_quant.QuantizationMethods)
                            and name not in overrides):
                        raise ValueError(
                            f"Quantization method {name} is an override but "
                            "is has not been added to the `overrides` list "
                            "above. This is necessary to ensure that the "
                            "overrides are checked in order of preference.")
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)

    def _verify_cuda_graph(self) -> None:
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)
        # CUDAGraph capture not supported for enc-dec models and mllama on ROCm
        ROCM_UNSUPPORTED_MODELS = ['mllama']
        unsupported_rocm = (self.hf_config.model_type
                            in ROCM_UNSUPPORTED_MODELS
                            or self.is_encoder_decoder)

        if (unsupported_rocm and not self.enforce_eager
                and current_platform.is_rocm()):
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.", self.hf_config.model_type)
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.46.1) with 8-bit models does not
        yet support CUDA graph.
        # TODO Remove this when bitsandbytes supports.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (getattr(self.hf_config,
                                           "quantization_config", None)
                                   is not None)
        is_8bit = (self.hf_config.quantization_config.get(
            "load_in_8bit", False) if has_quantization_config else False)
        if all([
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
        ]):
            logger.warning(
                "CUDA graph is not supported on BitsAndBytes 8bit yet, "
                "fallback to the eager mode.")

            self.enforce_eager = True

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")

    def verify_dual_chunk_attention_config(
        self,
        load_config: "LoadConfig",
    ) -> None:
        if hasattr(self.hf_config, "dual_chunk_attention_config"):
            # Try loading the sparse attention config
            from vllm.model_executor.model_loader.weight_utils import (
                get_sparse_attention_config)
            sparse_attn_config = get_sparse_attention_config(self, load_config)
            if sparse_attn_config:
                self.hf_config.dual_chunk_attention_config[
                    "sparse_attention_config"] = sparse_attn_config
                if "sparse_attention_enabled" not in \
                        self.hf_config.dual_chunk_attention_config:
                    self.hf_config.dual_chunk_attention_config[
                        "sparse_attention_enabled"] = True

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        from vllm.platforms import current_platform
        if not current_platform.is_async_output_supported(self.enforce_eager):
            self.use_async_output_proc = False
            return

        if envs.VLLM_USE_RAY_SPMD_WORKER:
            self.use_async_output_proc = False
            return

        # Async postprocessor is not necessary for pooling models
        # since there is no token generation
        if self.runner_type == "pooling":
            self.use_async_output_proc = False

        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        if speculative_config:
            self.use_async_output_proc = False

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:

        if parallel_config.distributed_executor_backend == "external_launcher":
            assert self.seed is not None, (
                "Seed must be set when using external launcher backend to "
                "make sure sampling results are the same across workers.")

        total_num_attention_heads = getattr(self.hf_text_config,
                                            "num_attention_heads", 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        if parallel_config.enable_expert_parallel:
            self._verify_with_expert_parallelism()

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            if not self.registry.is_pp_supported_model(self.architectures,
                                                       self):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                self.use_async_output_proc = False

    def get_hf_config_sliding_window(
            self) -> Union[Optional[int], list[Optional[int]]]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[Union[int, list[Optional[int]]]]:
        """Get the sliding window size, or None if disabled.
        """
        # If user disables sliding window, return None.
        if self.disable_sliding_window:
            return None
        # Otherwise get the value from the hf config.
        return self.get_hf_config_sliding_window()

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
            ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'kimi_k2'):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == 'eagle':
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return self.hf_text_config.model.model_type in \
                    ('deepseek_v2', 'deepseek_v3') \
                and self.hf_text_config.kv_lora_rank is not None
        return False

    def get_head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim",
                                       0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config,
                                           "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if hasattr(self.hf_text_config,
                   "model_type") and (self.hf_text_config.model_type
                                      == "zamba2"):
            return self.hf_text_config.attention_head_dim

        if self.is_attention_free:
            return 0

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # FIXME(woosuk): This may not be true for all models.
        return (self.hf_text_config.hidden_size //
                self.hf_text_config.num_attention_heads)

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_text_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(self.hf_config.attn_config, "kv_n_heads",
                           self.hf_config.num_attention_heads)

        if self.hf_config.model_type == "nemotron-nas":
            for block in self.hf_config.block_configs:
                if not block.attention.no_op:
                    return self.hf_config.num_attention_heads \
                        // block.attention.n_heads_in_group

            raise RuntimeError("Couldn't determine number of kv heads")

        if self.is_attention_free:
            return 0

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self,
                                parallel_config: "ParallelConfig") -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_layers_start_end_indices(
            self, parallel_config: "ParallelConfig") -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        if (self.hf_text_config.model_type == "deepseek_mtp"
                or self.hf_config.model_type == "mimo_mtp"
                or self.hf_config.model_type == "glm4_moe_mtp"):
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_nextn_predict_layers", 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_hidden_layers", 0)
        # the layout order is: DP x PP x TP
        pp_rank = (parallel_config.rank // parallel_config.tensor_parallel_size
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: "ParallelConfig",
        block_type: LayerBlockType = LayerBlockType.attention,
    ) -> int:
        # This function relies on 'layers_block_type' in hf_config,
        # for w/o this attribute, we will need to have workarounds like so
        attn_block_type = block_type == LayerBlockType.attention
        is_transformer = not self.is_hybrid and \
                            not self.has_noops and \
                            not self.is_attention_free
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            # Handle the basic case first
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            # Attention free
            # Note that this code assumes there
            # is only one type of attention-free block type.
            return 0 if attn_block_type else end - start
        elif self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op
                       for bc in block_configs[start:end])
        else:
            # Hybrid model Jamba
            layers_block_type_value = getattr(self.hf_config,
                                              "layers_block_type", None)
            if layers_block_type_value is not None:
                if hasattr(self.hf_text_config,
                           "model_type") and (self.hf_text_config.model_type
                                              == "zamba2"):
                    if attn_block_type:
                        return sum(t == "hybrid"
                                   for t in layers_block_type_value[start:end])
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type.value
                           for t in layers_block_type_value[start:end])

            # Hybrid model Minimax
            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])

            if layers_block_type_value is None and attn_type_list is None:
                raise ValueError(
                    "The model is an hybrid without a"
                    "layers_block_type or an attn_type_list in the hf_config,"
                    "cannot determine the num of "
                    f"{block_type.value} layers")

            return sum(t == 1 for t in attn_type_list[start:end])

    def get_mamba_chunk_size(self) -> Optional[int]:
        """
        Returns the mamba chunk size if it exists
        """
        # used by e.g. Bamba, FalconH1, Granite, PLaMo2
        chunk_size = getattr(self.hf_text_config, "mamba_chunk_size", None)
        if chunk_size is None:
            # used by e.g. Mamba2, NemotronH, Zamba
            chunk_size = getattr(self.hf_text_config, "chunk_size", None)
        return chunk_size

    def get_multimodal_config(self) -> "MultiModalConfig":
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(self) -> dict[str, Any]:
        if self.generation_config in ("auto", "vllm"):
            config = try_get_generation_config(
                self.hf_config_path or self.model,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
            )

        if config is None:
            return {}

        return config.to_diff_dict()

    def get_diff_sampling_param(self) -> dict[str, Any]:
        """
        This method returns a dictionary containing the parameters
        that differ from the default sampling parameters. If
        `generation_config` is `"vllm"`, an empty dictionary is returned.

        Returns:
            dict[str, Any]: A dictionary with the differing sampling
            parameters, if `generation_config` is `"vllm"` an empty dictionary.
        """
        if self.generation_config == "vllm":
            config = {}
        else:
            config = self.try_get_generation_config()

        # Overriding with given generation config
        config.update(self.override_generation_config)

        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p)
                for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens")
        else:
            diff_sampling_param = {}

        if diff_sampling_param:
            logger.warning_once(
                "Default sampling parameters have been overridden by the "
                "model's Hugging Face generation config recommended from the "
                "model creator. If this is not intended, please relaunch "
                "vLLM instance with `--generation-config vllm`.")
        return diff_sampling_param

    @property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        """
        For Mllama, VLLM overrides HF's is_encoder_decoder flag and sets it to
        True to enable cross-attention
        Neuron needs all multimodal data to be in the decoder and does not
        need to explicitly enable cross-attention
        """
        if (current_platform.is_neuron()
                and self.hf_config.model_type == "mllama"):
            return False

        return is_encoder_decoder(self.hf_config)

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_cross_encoder(self) -> bool:
        return (self._model_info.supports_cross_encoding
                or self.convert_type == "classify")

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_multimodal_raw_input_supported(self) -> bool:
        return self._model_info.supports_multimodal_raw_input

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        return self._model_info.is_hybrid

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def is_v1_compatible(self) -> bool:
        return not self._model_info.supports_v0_only

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def is_matryoshka(self) -> bool:
        return (bool(getattr(self.hf_config, "matryoshka_dimensions", None))
                or getattr(self.hf_config, "is_matryoshka", False))

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    @property
    def use_pad_token(self) -> bool:
        # cross_encoder models defaults to using pad_token.
        # `llm as reranker` models defaults to not using pad_token.
        return getattr(self.hf_config, "use_pad_token", True)

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        if (self.runner_type == "pooling" and getattr(
                self.hf_config, "position_embedding_type", "") == "absolute"):
            tokenizer_config = try_get_tokenizer_config(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                revision=self.tokenizer_revision)
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config)
        logger.info("Using max model len %s", max_model_len)
        return max_model_len


BlockSize = Literal[1, 8, 16, 32, 64, 128]
CacheDType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]
PrefixCachingHashAlgo = Literal["builtin", "sha256", "sha256_cbor_64bit"]


@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. This is ignored on
    neuron devices and set to `--max-model-len`. On CUDA devices, only block
    sizes up to 32 are supported. On HPU devices, block size defaults to 128.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform."""
    gpu_memory_utilization: float = 0.9
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = 4
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc)."""
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: Optional[int] = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: Optional[int] = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: Optional[bool] = None
    """Whether to enable prefix caching. Disabled by default for V0. Enabled by
    default for V1."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "builtin"
    """Set the hash algorithm for prefix caching:\n
    - "builtin" is Python's built-in hash.\n
    - "sha256" is collision resistant but with certain overheads.
    This option uses Pickle for object serialization before hashing.\n
    - "sha256_cbor_64bit" provides a reproducible, cross-language compatible 
    hash. It serializes objects using canonical CBOR and hashes them with 
    SHA-256. The resulting hash consists of the lower 64 bits of the SHA-256
    digest."""
    cpu_offload_gb: float = 0
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    cpu_kvcache_space_bytes: Optional[int] = None
    """(CPU backend only) CPU key-value cache space."""
    mamba_page_size_padded: Optional[int] = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""

    # Will be set after profiling.
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.cache_dtype)
        # `cpu_offload_gb` does not use `torch.compile` yet.
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        self.swap_space_bytes = self.swap_space * GiB_bytes

        self._verify_cache_dtype()
        self._verify_prefix_caching()

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.cpu_offload_gb < 0:
            raise ValueError("CPU offload space must be non-negative"
                             f", but got {self.cpu_offload_gb}")

        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

        return self

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in get_args(CacheDType):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor.")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None and not envs.VLLM_USE_V1:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

        if (self.enable_prefix_caching and self.prefix_caching_hash_algo
                not in get_args(PrefixCachingHashAlgo)):
            raise ValueError(
                "Unknown prefix caching hash algorithm: "
                f"{self.prefix_caching_hash_algo}. Must be one of "
                f"{get_args(PrefixCachingHashAlgo)}.")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
               f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
               "is allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)


@config
@dataclass
class LoadConfig:
    """Configuration for loading the model weights."""

    load_format: Union[str, LoadFormats] = "auto"
    """The format of the model weights to load:\n
    - "auto" will try to load the weights in the safetensors format and fall
    back to the pytorch bin format if safetensors format is not available.\n
    - "pt" will load the weights in the pytorch bin format.\n
    - "safetensors" will load the weights in the safetensors format.\n
    - "npcache" will load the weights in pytorch format and store a numpy cache
    to speed up the loading.\n
    - "dummy" will initialize the weights with random values, which is mainly
    for profiling.\n
    - "tensorizer" will use CoreWeave's tensorizer library for fast weight
    loading. See the Tensorize vLLM Model script in the Examples section for
    more information.\n
    - "runai_streamer" will load the Safetensors weights using Run:ai Model
    Streamer.\n
    - "bitsandbytes" will load the weights using bitsandbytes quantization.\n
    - "sharded_state" will load weights from pre-sharded checkpoint files,
    supporting efficient loading of tensor-parallel models.\n
    - "gguf" will load weights from GGUF format files (details specified in
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).\n
    - "mistral" will load weights from consolidated safetensors files used by
    Mistral models.
    - Other custom values can be supported via plugins."""
    download_dir: Optional[str] = None
    """Directory to download and load the weights, default to the default
    cache directory of Hugging Face."""
    model_loader_extra_config: Union[dict, TensorizerConfig] = field(
        default_factory=dict)
    """Extra config for model loader. This will be passed to the model loader
    corresponding to the chosen load_format."""
    device: Optional[str] = None
    """Device to which model weights will be loaded, default to
    device_config.device"""
    ignore_patterns: Optional[Union[list[str], str]] = None
    """The list of patterns to ignore when loading the model. Default to
    "original/**/*" to avoid repeated loading of llama's checkpoints."""
    use_tqdm_on_load: bool = True
    """Whether to enable tqdm for showing progress bar when loading model
    weights."""
    pt_load_map_location: Union[str, dict[str, str]] = "cpu"
    """
    pt_load_map_location: the map location for loading pytorch checkpoint, to
    support loading checkpoints can only be loaded on certain devices like
    "cuda", this is equivalent to {"": "cuda"}. Another supported format is
    mapping from different devices like from GPU 1 to GPU 0:
    {"cuda:1": "cuda:0"}. Note that when passed from command line, the strings
    in dictionary needs to be double quoted for json parsing. For more details,
    see original doc for `map_location` in https://pytorch.org/docs/stable/generated/torch.load.html
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        self.load_format = self.load_format.lower()
        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]


DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]


@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: Optional[int] = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: str = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kuberentes. Set implicitly when --data-parallel-rank
    is provided explicitly to vllm serve."""
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with 
    --data-parallel-start-rank."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    num_redundant_experts: int = 0
    """Number of redundant experts to use for expert parallelism."""
    eplb_window_size: int = 1000
    """Window size for expert load recording."""
    eplb_step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `eplb_window_size` steps will be used for rearranging experts.
    """
    eplb_log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """

    max_parallel_loading_workers: Optional[int] = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    placement_group: Optional["PlacementGroup"] = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: Optional[Union[DistributedExecutorBackend,
                                                 type["ExecutorBase"]]] = None
    """Backend to use for distributed model
    workers, either "ray" or "mp" (multiprocessing). If the product
    of pipeline_parallel_size and tensor_parallel_size is less than
    or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, this will default
    to "ray" if Ray is installed and fail otherwise. Note that tpu
    only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""

    world_size: int = field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    enable_multimodal_encoder_data_parallel: bool = False
    """ Use data parallelism instead of tensor parallelism for vision encoder.
    Only support LLama4 for now"""

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        increment the port number each time we need to initialize a
        new process group related to data parallelism.
        """
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += 1
        return answer

    def stateless_init_dp_group(self) -> "ProcessGroup":
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group)

        max_retries = 5
        last_exc: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo")
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning(
                        "Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def has_unfinished_dp(dp_group: "ProcessGroup",
                          has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished],
                              dtype=torch.int32,
                              device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: "ProcessGroup",
                                  kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory],
                              dtype=torch.int64,
                              device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        factors.append(self.enable_expert_parallel)
        factors.append(self.data_parallel_size)
        factors.append(envs.VLLM_ALL2ALL_BACKEND)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})")

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            self.data_parallel_master_port = get_open_port()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})")
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_external_lb:
                raise ValueError("data_parallel_external_lb can only "
                                 "be set when data_parallel_size > 1")

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.enable_eplb:
            if not current_platform.is_cuda():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices now.")
            if self.num_redundant_experts < 0:
                raise ValueError(
                    "num_redundant_experts must be non-negative, but got "
                    f"{self.num_redundant_experts}.")
            if not self.enable_expert_parallel:
                raise ValueError(
                    "enable_expert_parallel must be True to use EPLB.")
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            if self.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts should be used with EPLB."
                    f"{self.num_redundant_experts}.")
        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray: "
                                     f"{ray_utils.ray_import_err}. Ray is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.")
                backend = "ray"
            elif self.data_parallel_backend == "ray":
                logger.info("Using ray distributed inference because "
                            "data_parallel_backend is ray")
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference",
                         backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and self.distributed_executor_backend.uses_ray)

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase
        from vllm.platforms import current_platform
        if self.distributed_executor_backend not in (
                "ray", "mp", "uni",
                "external_launcher", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher' or"
                " custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")

        return self


PreemptionMode = Literal["swap", "recompute"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: SkipValidation[int] = None  # type: ignore
    """Maximum number of tokens to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_num_seqs: SkipValidation[int] = None  # type: ignore
    """Maximum number of sequences to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_model_len: SkipValidation[int] = None  # type: ignore
    """Maximum length of a sequence (including prompt and generated text). This
    is primarily set in `ModelConfig` and that value should be manually
    duplicated here."""

    max_num_partial_prefills: int = 1
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = 1
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    num_lookahead_slots: int = 0
    """The number of slots to allocate per sequence per
    step, beyond the known token ids. This is used in speculative
    decoding to store KV activations of tokens which may or may not be
    accepted.

    NOTE: This will be replaced by speculative config in the future; it is
    present to enable correctness tests until then."""

    cuda_graph_sizes: list[int] = field(default_factory=list)
    """Cuda graph capture sizes
    1. if none provided, then default set to [min(max_num_seqs * 2, 512)]
    2. if one value is provided, then the capture list would follow the
    pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]
    3. more than one value (e.g. 1 2 128) is provided, then the capture list
    will follow the provided list."""

    delay_factor: float = 0.0
    """Apply a delay (of delay factor multiplied by previous
    prompt latency) before scheduling next prompt."""

    enable_chunked_prefill: SkipValidation[bool] = None  # type: ignore
    """If True, prefill requests can be chunked based
    on the remaining max_num_batched_tokens."""

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    preemption_mode: Optional[PreemptionMode] = None
    """Whether to perform preemption by swapping or
    recomputation. If not specified, we determine the mode as follows:
    We use recomputation by default since it incurs lower overhead than
    swapping. However, when the sequence group has multiple sequences
    (e.g., beam search), recomputation is not currently supported. In
    such a case, we use swapping instead."""

    num_scheduler_steps: int = 1
    """Maximum number of forward steps per scheduler call."""

    multi_step_stream_outputs: bool = True
    """If False, then multi-step will stream outputs at the end of all steps"""

    send_delta_data: bool = False
    """Private API. If used, scheduler sends delta data to
    workers instead of an entire data. It should be enabled only
    when SPMD worker architecture is enabled. I.e.,
    VLLM_USE_RAY_SPMD_WORKER=1"""

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:\n
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.\n
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""

    chunked_prefill_enabled: bool = field(init=False)
    """True if chunked prefill is enabled."""

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    # scheduler class or path. "vllm.core.scheduler.Scheduler" (default)
    # or "mod.custom_class".
    scheduler_cls: Union[str, type[object]] = "vllm.core.scheduler.Scheduler"
    """The scheduler class to use. "vllm.core.scheduler.Scheduler" is the
    default scheduler. Can be a class directly or the path to a class of form
    "mod.custom_class"."""

    disable_hybrid_kv_cache_manager: bool = False
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    """

    async_scheduling: bool = False
    """EXPERIMENTAL: If set to True, perform async scheduling. This may help
    reduce the CPU overheads, leading to better latency and throughput. However,
    async scheduling is currently not supported with some features such as
    structured outputs, speculative decoding, and pipeline parallelism.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.max_model_len is None:
            self.max_model_len = 8192

        if self.max_num_seqs is None:
            self.max_num_seqs = 128

        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(
                        self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
                else:
                    self.max_num_batched_tokens = (
                        DEFAULT_MAX_NUM_BATCHED_TOKENS)
            else:
                # If max_model_len is too short, use
                # DEFAULT_MAX_NUM_BATCHED_TOKENS as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(
                    self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * self.max_model_len,
                self.max_num_batched_tokens)

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(self.max_model_len *
                                                        0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills, self.max_long_partial_prefills,
                self.long_prefill_token_threshold)

        # NOTE: Default set cuda_graph_sizes to [min(max_num_seqs * 2, 512)].
        # This avoids OOM in tight memory scenarios with small max_num_seqs,
        # and prevents capture of many large graphs (>512) that would greatly
        # increase startup time with limited performance benefit.
        if not self.cuda_graph_sizes:
            self.cuda_graph_sizes = [min(self.max_num_seqs * 2, 512)]

        if self.async_scheduling:
            self.scheduler_cls = (
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler")

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.max_num_batched_tokens > self.max_num_seqs * self.max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * self.max_model_len)

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
                "must be greater than or equal to 1.")
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError("Chunked prefill must be enabled to set "
                                 "max_num_partial_prefills > 1.")

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({self.max_model_len}).")

        if (self.max_long_partial_prefills
                < 1) or (self.max_long_partial_prefills
                         > self.max_num_partial_prefills):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                "must be greater than or equal to 1 and less than or equal to "
                f"max_num_partial_prefills ({self.max_num_partial_prefills}).")

        return self

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1


Device = Literal["auto", "cuda", "neuron", "cpu", "tpu", "xpu"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    device: SkipValidation[Optional[Union[Device, torch.device]]] = "auto"
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # the device/platform information will be summarized
        # by torch/vllm automatically.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            from vllm.platforms import current_platform
            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue.")
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron"]:
            self.device = torch.device("cpu")
        elif self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


SpeculativeMethod = Literal["ngram", "eagle", "eagle3", "medusa",
                            "mlp_speculator", "draft_model", "deepseek_mtp"]


@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # General speculative decoding control
    num_speculative_tokens: SkipValidation[int] = None  # type: ignore
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: Optional[str] = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: Optional[SpeculativeMethod] = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: Optional[int] = None
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    disable_logprobs: bool = True
    """If set to True, token log probabilities are not returned during
    speculative decoding. If set to False, token log probabilities are returned
    according to the log probability settings in SamplingParams."""

    # Draft model configuration
    quantization: Optional[me_quant.QuantizationMethods] = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: Optional[int] = None
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: Optional[str] = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_by_batch_size: Optional[int] = None
    """Disable speculative decoding for new incoming requests when the number
    of enqueued requests is larger than this value, if provided."""

    # Ngram proposer configuration
    prompt_lookup_max: Optional[int] = None
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: Optional[int] = None
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    speculative_token_tree: Optional[str] = None
    """Specifies the tree structure for speculative token generation.
    """
    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[
        ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""
    enable_chunked_prefill: SkipValidation[bool] = None  # type: ignore
    """Whether vLLM is configured to use chunked prefill or not. Used for
    raising an error since it's not yet compatible with speculative decode."""
    disable_log_stats: SkipValidation[bool] = None  # type: ignore
    """Whether to disable the periodic printing of stage times in speculative
    decoding."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[
        ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 affects the computation graph because it returns intermediate
        # hidden states in addition to the final hidden state.
        factors.append(self.method == "eagle3")
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> "SpeculativeConfig":
        """Parse the CLI value for the speculative config."""
        return cls(**dict_value)

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type == "deepseek_v3":
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"]
            })

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["Glm4MoeMTPModel"]
            })

        return hf_config

    def __post_init__(self):

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                (self.target_model_config.hf_text_config.model_type \
                        == "deepseek_v3" or
                    self.target_model_config.hf_text_config.model_type \
                        == "mimo"):
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                elif (self.draft_model_config.hf_config.model_type
                      in ("deepseek_mtp", "mimo_mtp", "glm4_moe_mtp")):
                    self.method = "deepseek_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                                "All Deepseek MTP models only have " \
                                "one layer. Might need some code changes " \
                                "to support multiple layers."
                            )
                else:
                    self.method = "draft_model"
                    raise NotImplementedError(
                        "Speculative decoding with draft model is not "
                        "supported yet. Please consider using other "
                        "speculative decoding methods such as ngram, medusa, "
                        "eagle, or deepseek_mtp.")

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)
                    if isinstance(self.draft_model_config.hf_config,
                                  EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle")
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                    self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: Optional[int],
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:

            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {draft_max_model_len=}")

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {target_max_model_len=}")

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
            target_parallel_config: ParallelConfig,
            speculative_draft_tensor_parallel_size: Optional[int],
            draft_hf_config: PretrainedConfig) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type)
            else:
                speculative_draft_tensor_parallel_size = \
                    target_parallel_config.tensor_parallel_size
        elif speculative_draft_tensor_parallel_size not in (
                1, target_parallel_config.tensor_parallel_size):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size")
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.
            pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.
            distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.
            max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.
            disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.
            ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter.")

        if self.num_speculative_tokens <= 0:
            raise ValueError("Expected num_speculative_tokens to be greater "
                             f"than zero ({self.num_speculative_tokens}).")

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config)

        if (self.disable_by_batch_size is not None
                and self.disable_by_batch_size < 2):
            raise ValueError("Expect the batch size threshold of disabling "
                             "speculative decoding is > 1, but got "
                             f"{self.disable_by_batch_size=}")

        if self.method == "eagle3" and self.target_model_config and \
            "llama" not in self.target_model_config.hf_text_config.model_type:
            raise ValueError(
                "Eagle3 is only supported for Llama models. "
                f"Got {self.target_model_config.hf_text_config.model_type=}")

        return self

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "deepseek_mtp")

    def __repr__(self) -> str:
        method = self.method
        model = None if method == "ngram" else self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"


LoRADType = Literal["auto", "float16", "bfloat16"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: int = 16
    """Max LoRA rank."""
    max_loras: int = 1
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: Optional[int] = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: Union[torch.dtype, LoRADType] = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    lora_extra_vocab_size: int = 256
    """Maximum size of extra vocabulary that can be present in a LoRA adapter
    (added to the base model vocabulary)."""
    lora_vocab_padding_size: ClassVar[int] = current_platform\
        .get_lora_vocab_padding_size()

    default_mm_loras: Optional[dict[str, str]] = None
    """Dictionary mapping specific modalities to LoRA model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a LoRA to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own LoRA, we do NOT apply
    default_mm_loras because we currently only support one lora adapter
    per prompt. When run in offline mode, the lora IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""
    bias_enabled: bool = False
    """Enable bias for LoRA adapters."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.lora_extra_vocab_size)
        factors.append(self.lora_vocab_padding_size)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # Setting the maximum rank to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_cache_config(self, cache_config: CacheConfig):
        if cache_config.cpu_offload_gb > 0 and not envs.VLLM_USE_V1:
            raise ValueError(
                "V0 LoRA does not support CPU offload, please use V1.")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)


@config
@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: dict[str, int] = \
        cast(dict[str, int], get_field(ModelConfig, "limit_mm_per_prompt"))
    """
    The maximum number of input items allowed per prompt for each modality.
    Defaults to 1 (V0) or 999 (V1) for each modality.

    For example, to allow up to 16 images and 2 videos per prompt:
    `{"image": 16, "video": 2}`
    """

    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Additional args passed to process media inputs, keyed by modalities.
    For example, to set num_frames for video, set
    `--media-io-kwargs '{"video": {"num_frames": 40} }'` """

    mm_processor_kwargs: Optional[dict[str, object]] = None
    """
    Overrides for the multi-modal processor obtained from
    `transformers.AutoProcessor.from_pretrained`.

    The available overrides depend on the model that is being run.

    For example, for Phi-3-Vision:
    `{"num_crops": 4}`.
    """

    disable_mm_preprocessor_cache: bool = False
    """
    If `True`, disable caching of the processed multi-modal inputs.
    """

    interleave_mm_strings: bool = False
    """
    Enable fully interleaved support for multimodal prompts.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def get_limit_per_prompt(self, modality: str) -> int:
        """
        Get the maximum number of input items allowed per prompt
        for the given modality.
        """
        return self.limit_per_prompt.get(
            modality,
            999 if envs.VLLM_USE_V1 else 1,
        )

    # TODO: Add configs to init vision tower or not.


@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: Optional[str] = None
    """
    The pooling method of the pooling model. This should be a key in
    [`vllm.model_executor.layers.pooler.PoolingType`][].
    """

    normalize: Optional[bool] = None
    """
    Whether to normalize the pooled outputs. Usually, this should be set to
    ``True`` for embedding outputs.
    """

    softmax: Optional[bool] = None
    """
    Whether to apply softmax to the pooled outputs. Usually, this should be set
    to ``True`` for classification outputs.
    """

    step_tag_id: Optional[int] = None
    """
    If set, only the score corresponding to the ``step_tag_id`` in the
    generated sentence should be returned. Otherwise, the scores for all tokens
    are returned.
    """

    returned_token_ids: Optional[list[int]] = None
    """
    A list of indices for the vocabulary dimensions to be extracted,
    such as the token IDs of ``good_token`` and ``bad_token`` in the
    ``math-shepherd-mistral-7b-prm`` model.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# model_type -> reason
_FLOAT16_NOT_SUPPORTED_MODELS = {
    "gemma2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3": "Numerical instability. Please use bfloat16 or float32 instead.",
    "plamo2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "glm4": "Numerical instability. Please use bfloat16 or float32 instead.",
}


def _is_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:  # noqa: E501, SIM103
        return False

    return True


def _check_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        reason = _FLOAT16_NOT_SUPPORTED_MODELS[model_type]
        raise ValueError(f"The model type {model_type!r} "
                         f"does not support float16. Reason: {reason}")

    return True


def _find_dtype(
    model_id: str,
    config: PretrainedConfig,
    *,
    revision: Optional[str],
):
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define torch_dtype
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), "torch_dtype", None)
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "torch_dtype", None)
    if config_dtype is None and hasattr(config, "encoder_config"):
        config_dtype = getattr(config.encoder_config, "torch_dtype", None)

    # Try to read the dtype of the weights if they are in safetensors format
    if config_dtype is None:
        repo_mt = try_get_safetensors_metadata(model_id, revision=revision)

        if repo_mt and (files_mt := repo_mt.files_metadata):
            param_dtypes: set[torch.dtype] = {
                _SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
                for file_mt in files_mt.values()
                for dtype_str in file_mt.parameter_count
                if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE
            }

            if param_dtypes:
                return common_broadcastable_dtype(param_dtypes)

    if config_dtype is None:
        config_dtype = torch.float32

    return config_dtype


def _resolve_auto_dtype(
    model_type: str,
    config_dtype: torch.dtype,
    *,
    is_pooling_model: bool,
):
    from vllm.platforms import current_platform

    supported_dtypes = [
        dtype for dtype in current_platform.supported_dtypes
        if _is_valid_dtype(model_type, dtype)
    ]

    if is_pooling_model and torch.float16 in supported_dtypes:
        preferred_dtype = torch.float16
    else:
        preferred_dtype = supported_dtypes[0]

    # Downcast for float32 models
    if config_dtype == torch.float32:
        config_dtype = preferred_dtype

    if config_dtype in supported_dtypes:
        return config_dtype

    # Ensure device compatibility
    device_name = current_platform.get_device_name()
    device_capability = current_platform.get_device_capability()

    if device_capability is None:
        device_str = f"{device_name!r}"
    else:
        version_str = device_capability.as_version_str()
        device_str = f"{device_name!r} (with compute capability {version_str})"

    logger.warning(
        "Your device %s doesn't support %s. "
        "Falling back to %s for compatibility.",
        device_str,
        config_dtype,
        preferred_dtype,
    )

    return preferred_dtype


def _get_and_verify_dtype(
    model_id: str,
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
    *,
    is_pooling_model: bool,
    revision: Optional[str] = None,
) -> torch.dtype:
    config_dtype = _find_dtype(model_id, config, revision=revision)
    model_type = config.model_type

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            torch_dtype = _resolve_auto_dtype(
                model_type,
                config_dtype,
                is_pooling_model=is_pooling_model,
            )
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype!r}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    _check_valid_dtype(model_type, torch_dtype)

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    tokenizer_config: Optional[dict],
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window_len: Optional[Union[int, list[Optional[int]]]],
    spec_target_max_model_len: Optional[int] = None,
    encoder_config: Optional[Any] = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    # For Command-R / Cohere, Cohere2 / Aya Vision models
    if tmp_max_len := getattr(hf_config, "model_max_length", None):
        max_len_key = "model_max_length"
        derived_max_model_len = tmp_max_len

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:

        sliding_window_len_min = get_min_sliding_window(sliding_window_len)
        max_len_key = "sliding_window" \
            if sliding_window_len_min < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len,
                                    sliding_window_len_min)

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len)
        derived_max_model_len = min(derived_max_model_len,
                                    tokenizer_model_max_length)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        if spec_target_max_model_len is not None:
            # If this is a speculative draft model, we use the max model len
            # from the target model.
            return spec_target_max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.", possible_keys,
            default_max_len)
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    # NOTE(woosuk): Gemma3's max_model_len (128K) is already scaled by RoPE
    # scaling, so we skip applying the scaling factor again.
    if rope_scaling is not None and "gemma3" not in hf_config.model_type:
        # No need to consider "type" key because of patch_rope_scaling when
        # loading HF config
        rope_type = rope_scaling["rope_type"]

        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            # NOTE: rope_type == "default" does not define factor
            # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
            scaling_factor = rope_scaling.get("factor", 1.0)

            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
        if current_platform.is_tpu():
            logger.warning(
                "--max-model-len is not specified, "
                "it's currently using model's default length %s, "
                "which might be too large."
                "Please input with --max-model-len based on your "
                "request input length and output length, to avoid "
                "unnecessary degradation.", max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate.")
        else:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors.")
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning(
                    "%s Make sure the value is correct and within the "
                    "model context size.", msg)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    "the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")
    return int(max_model_len)


def get_min_sliding_window(
        sliding_window: Union[int, list[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)

    return sliding_window


def get_served_model_name(model: str,
                          served_model_name: Optional[Union[str, list[str]]]):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


GuidedDecodingBackendV0 = Literal["auto", "outlines", "lm-format-enforcer",
                                  "xgrammar", "guidance"]

GuidedDecodingBackendV1 = Literal["auto", "xgrammar", "guidance", "outlines"]
GuidedDecodingBackend = Literal[GuidedDecodingBackendV0,
                                GuidedDecodingBackendV1]


@config
@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine."""

    backend: GuidedDecodingBackend = "auto" if envs.VLLM_USE_V1 else "xgrammar"
    """Which engine will be used for guided decoding (JSON schema / regex etc)
    by default. With "auto", we will make opinionated choices based on request
    contents and what the backend libraries currently support, so the behavior
    is subject to change in each release."""

    disable_fallback: bool = False
    """If `True`, vLLM will not fallback to a different backend on error."""

    disable_any_whitespace: bool = False
    """If `True`, the model will not generate any whitespace during guided
    decoding. This is only supported for xgrammar and guidance backends."""

    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""

    reasoning_backend: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if envs.VLLM_USE_V1:
            valid_guided_backends = get_args(GuidedDecodingBackendV1)
        else:
            valid_guided_backends = get_args(GuidedDecodingBackendV0)
        if self.backend not in valid_guided_backends:
            raise ValueError(f"Invalid backend '{self.backend}',"
                             f" must be one of {valid_guided_backends}")
        if (self.disable_any_whitespace
                and self.backend not in ("xgrammar", "guidance")):
            raise ValueError("disable_any_whitespace is only supported for "
                             "xgrammar and guidance backends.")
        if (self.disable_additional_properties and self.backend != "guidance"):
            raise ValueError("disable_additional_properties is only supported "
                             "for the guidance backend.")


DetailedTraceModules = Literal["model", "worker", "all"]


@config
@dataclass
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    show_hidden_metrics_for_version: Optional[str] = None
    """Enable deprecated Prometheus metrics that have been hidden since the
    specified version. For example, if a previously deprecated metric has been
    hidden since the v0.7.0 release, you use
    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
    you migrate to new metrics. The metric is likely to be removed completely
    in an upcoming release."""

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        if self.show_hidden_metrics_for_version is None:
            return False
        return version._prev_minor_version_was(
            self.show_hidden_metrics_for_version)

    otlp_traces_endpoint: Optional[str] = None
    """Target URL to which OpenTelemetry traces will be sent."""

    collect_detailed_traces: Optional[list[DetailedTraceModules]] = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        return (self.collect_detailed_traces is not None
                and ("model" in self.collect_detailed_traces
                     or "all" in self.collect_detailed_traces))

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        return (self.collect_detailed_traces is not None
                and ("worker" in self.collect_detailed_traces
                     or "all" in self.collect_detailed_traces))

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if (self.collect_detailed_traces is not None
                and len(self.collect_detailed_traces) == 1
                and "," in self.collect_detailed_traces[0]):
            self._parse_collect_detailed_traces()

        from vllm.tracing import is_otel_available, otel_import_error_traceback
        if not is_otel_available() and self.otlp_traces_endpoint is not None:
            raise ValueError(
                "OpenTelemetry is not available. Unable to configure "
                "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                f"installed. Original error:\n{otel_import_error_traceback}")

    def _parse_collect_detailed_traces(self):
        assert isinstance(self.collect_detailed_traces, list)
        self.collect_detailed_traces = cast(
            list[DetailedTraceModules],
            self.collect_detailed_traces[0].split(","))


KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


@config
@dataclass
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: Optional[str] = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: Optional[str] = None
    """The engine id for KV transfers."""

    kv_buffer_device: Optional[str] = "cuda"
    """The device used by kv connector to buffer the KV cache.
    Currently only support 'cuda'."""

    kv_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    kv_role: Optional[KVRole] = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    kv_rank: Optional[int] = None
    """The rank of this vLLM instance in the KV cache transfer. Typical value:
    0 for prefill instance, 1 for decode instance.
    Currently only 1P1D is supported."""

    kv_parallel_size: int = 1
    """The number of parallel instances for KV cache transfer. For
    PyNcclConnector, this should be 2."""

    kv_ip: str = "127.0.0.1"
    """The KV connector ip, used to build distributed connection."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: Optional[str] = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(f"Unsupported kv_role: {self.kv_role}. "
                             f"Supported roles are {get_args(KVRole)}")

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError("Please specify kv_disagg_role when kv_connector "
                             f"is set, supported roles are {get_args(KVRole)}")

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)


@config
@dataclass
class KVEventsConfig:
    """Configuration for KV event publishing."""

    enable_kv_cache_events: bool = False
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: Optional[str] = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """


class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


@config
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""

    enable_fusion: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    """Whether to enable the custom fusion (RMSNorm/SiluMul+quant) pass."""
    enable_attn_fusion: bool = False
    """Whether to enable the custom attention+quant fusion pass."""
    enable_noop: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    """Whether to enable the custom no-op elimination pass."""
    enable_sequence_parallelism: bool = False
    """Whether to enable sequence parallelism."""
    enable_async_tp: bool = False
    """Whether to enable async TP."""
    enable_fi_allreduce_fusion: bool = False
    """Whether to enable flashinfer allreduce fusion."""
    fi_allreduce_fusion_max_token_num: int = 1024
    """Max number of tokens to used in flashinfer allreduce fusion."""

    # TODO(luka) better pass enabling system.

    def uuid(self):
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        return InductorPass.hash_dict(asdict(self))

    def __post_init__(self) -> None:
        if not self.enable_noop:
            if self.enable_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work")
            if self.enable_attn_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work")


@config
@dataclass
class CompilationConfig:
    """Configuration for compilation. It has three parts:

    - Top-level Compilation control:
        - [`level`][vllm.config.CompilationConfig.level]
        - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
        - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
        - [`backend`][vllm.config.CompilationConfig.backend]
        - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
        - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
    - CudaGraph capture:
        - [`use_cudagraph`][vllm.config.CompilationConfig.use_cudagraph]
        - [`cudagraph_capture_sizes`]
        [vllm.config.CompilationConfig.cudagraph_capture_sizes]
        - [`cudagraph_num_of_warmups`]
        [vllm.config.CompilationConfig.cudagraph_num_of_warmups]
        - [`cudagraph_copy_inputs`]
        [vllm.config.CompilationConfig.cudagraph_copy_inputs]
        - [`full_cuda_graph`][vllm.config.CompilationConfig.full_cuda_graph]
    - Inductor compilation:
        - [`use_inductor`][vllm.config.CompilationConfig.use_inductor]
        - [`compile_sizes`][vllm.config.CompilationConfig.compile_sizes]
        - [`inductor_compile_config`]
        [vllm.config.CompilationConfig.inductor_compile_config]
        - [`inductor_passes`][vllm.config.CompilationConfig.inductor_passes]
        - custom inductor passes

    Why we have different sizes for cudagraph and inductor:
    - cudagraph: a cudagraph captured for a specific size can only be used
        for the same size. We need to capture all the sizes we want to use.
    - inductor: a graph compiled by inductor for a general shape can be used
        for different sizes. Inductor can also compile for specific sizes,
        where it can have more information to optimize the graph with fully
        static shapes. However, we find the general shape compilation is
        sufficient for most cases. It might be beneficial to compile for
        certain small batchsizes, where inductor is good at optimizing.
    """
    # Top-level Compilation control
    level: int = 0
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    debug_dump_path: str = ""
    """The path to dump the debug information."""
    cache_dir: str = ""
    """The directory to store the compiled graph, to accelerate Inductor
    compilation. By default, it will use model-related information to generate
    a cache directory."""
    backend: str = ""
    """The backend for compilation. It needs to be a string:

    - "" (empty string): use the default backend.
    - "eager"/"openxla"/...: use the specified backend registered in PyTorch.
    - "full.module.name": a qualified name which can be used to import the

    backend function.
    We use string to avoid serialization issues when using compilation in a
    distributed setting. When the compilation level is 1 or 2, the backend is
    used for the compilation directly (it sees the whole graph). When the
    compilation level is 3, the backend is used for the piecewise compilation
    (it sees a part of the graph)."""
    custom_ops: list[str] = field(default_factory=list)
    """Fine-grained control over which custom ops to enable/disable. Use 'all'
    to enable all, 'none' to disable all. Also specify a list of custom op
    names to enable (prefixed with a '+'), or disable (prefixed with a '-').
    Examples:

    - 'all,-op1' to enable all except op1
    - 'none,+op1,+op2' to enable only op1 and op2

    By default, all custom ops are enabled when running without Inductor and
    disabled when running with Inductor: level>=PIECEWISE and use_inductor=True.
    Inductor generates (fused) Triton kernels for disabled custom ops."""
    splitting_ops: list[str] = field(default_factory=list)
    """A list of ops to split the full graph into subgraphs, used in piecewise
    compilation."""

    # Inductor capture
    use_inductor: bool = True
    """Whether to use inductor compilation:

    - False: inductor compilation is not used. graph runs in eager
        (custom_ops enabled by default).
    - True: inductor compilation is used (custom_ops disabled by default).
        One graph for symbolic shape and one graph per size in compile_sizes
        are compiled using configurations in inductor_compile_config.

    This setting is ignored if level<PIECEWISE."""
    compile_sizes: Optional[list[Union[int, str]]] = None
    """Sizes to compile for inductor. In addition
    to integers, it also supports "cudagraph_capture_sizes" to
    specify the sizes for cudagraph capture."""
    inductor_compile_config: dict = field(default_factory=dict)
    """Additional configurations for inductor.
    - None: use default configurations."""
    inductor_passes: dict[str, str] = field(default_factory=dict)
    """Additional passes for inductor. It is a dictionary
    from pass name to pass function qualified name. We use function
    name because the config uses JSON format. If we pass the config
    from Python, functions can also be passed directly via Python object
    constructor, e.g. `CompilationConfig(inductor_passes={"a": func})`."""

    # CudaGraph compilation
    use_cudagraph: bool = field(default_factory=lambda: envs.VLLM_USE_V1)
    """Whether to use cudagraph inside compilation.
    - False: cudagraph inside compilation is not used.
    - True: cudagraph inside compilation is used. It requires
        that all input buffers have fixed addresses, and all
        splitting ops write their outputs to input buffers.
    In the vLLM V1 Engine, this flag only applies for
    CompilationLevel.PIECEWISE (aka -O3).
    Note that this is orthogonal to the cudagraph capture logic
    outside of compilation.
    TODO: move outside cudagraph logic into compilation.
    torch.compile will handle cudagraph capture logic in the future."""
    cudagraph_num_of_warmups: int = 0
    """Number of warmup runs for cudagraph.
    It means the first several runs will be treated as warmup runs.
    Only after that, the execution will be recorded, and the recorded
    cudagraph will be used for subsequent runs."""
    cudagraph_capture_sizes: Optional[list[int]] = None
    """Sizes to capture cudagraph.
    - None (default): capture sizes are inferred from vllm config.
    - list[int]: capture sizes are specified as given."""
    cudagraph_copy_inputs: bool = False
    """Whether to copy input tensors for
    cudagraph. If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True, and the compiler will copy the input to an
    internally managed buffer. Default is False."""
    full_cuda_graph: bool = False
    """whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs. Thus this
    flag cannot be used together with splitting_ops. This may provide
    performance benefits for smaller models."""

    pass_config: PassConfig = field(default_factory=PassConfig)
    """Custom inductor passes, see PassConfig for more details"""

    max_capture_size: int = field(default=None, init=False)  # type: ignore
    """not configurable, computed after init"""
    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    """local cache dir for each rank"""
    bs_to_padded_graph_size: list[int] = field(
        default=None,  # type: ignore
        init=False)
    """optimization:
    Intuitively, bs_to_padded_graph_size should be dict[int, int].
    since we know all keys are in a range [0, max_capture_size],
    we can optimize it to list[int] for better lookup performance."""

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = field(default_factory=Counter,
                                             init=False)
    """custom ops that are enabled"""
    disabled_custom_ops: Counter[str] = field(default_factory=Counter,
                                              init=False)
    """custom ops that are disabled"""
    traced_files: set[str] = field(default_factory=set, init=False)
    """files that are traced for compilation"""
    compilation_time: float = field(default=0.0, init=False)
    """time taken for compilation"""

    static_forward_context: dict[str, Any] = field(default_factory=dict,
                                                   init=False)
    """Per-model forward context
    Map from layer name to layer objects that need to be accessed outside
    model code, e.g., Attention, FusedMOE when dp_size>1."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.level)
        factors.append(self.backend)
        factors.append(self.custom_ops)
        factors.append(self.splitting_ops)
        factors.append(self.use_inductor)
        factors.append(self.inductor_compile_config)
        factors.append(self.inductor_passes)
        factors.append(self.pass_config.uuid())
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __repr__(self) -> str:
        exclude = {
            "static_forward_context": True,
            "enabled_custom_ops": True,
            "disabled_custom_ops": True,
            "compilation_time": True,
            "bs_to_padded_graph_size": True,
            "pass_config": True,
            "traced_files": True,
            "inductor_compile_config": {
                "post_grad_custom_post_pass": True,
            },
        }
        # The cast to string is necessary because Pydantic is mocked in docs
        # builds and sphinx-argparse doesn't know the return type of decode()
        return str(
            TypeAdapter(CompilationConfig).dump_json(
                self,
                exclude=exclude,  # type: ignore[arg-type]
                exclude_unset=True).decode())

    __str__ = __repr__

    @classmethod
    def from_cli(cls, cli_value: str) -> "CompilationConfig":
        """Parse the CLI value for the compilation config.
        -O1, -O2, -O3, etc. is handled in FlexibleArgumentParser.
        """
        return TypeAdapter(CompilationConfig).validate_json(cli_value)

    def __post_init__(self) -> None:
        count_none = self.custom_ops.count("none")
        count_all = self.custom_ops.count("all")
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"

        # TODO(zou3519/luka): There are 2 issues with auto-functionalization V2:
        # 1. A bug in PyTorch, fixed in 2.7:
        #    https://github.com/pytorch/pytorch/issues/147924
        # 2. Custom passes (fusion) rely on auto-functionalization V1 and don't
        #    work with V2. Addressing this will take extra engineering effort
        #    and it is not yet a priority. RFC here:
        #    https://github.com/vllm-project/vllm/issues/14703

        if is_torch_equal_or_newer("2.6"):
            KEY = 'enable_auto_functionalized_v2'
            if KEY not in self.inductor_compile_config:
                self.inductor_compile_config[KEY] = False

        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), (
                    f"pass {k} should be callable or a qualified name")
                self.inductor_compile_config[k] = v if isinstance(
                    v, InductorPass) else CallableInductorPass(v)
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func if isinstance(
                func, InductorPass) else CallableInductorPass(func)

        if isinstance(self.pass_config, dict):
            self.pass_config = PassConfig(**self.pass_config)

    def init_backend(self, vllm_config: "VllmConfig") -> Union[str, Callable]:
        if self.level == CompilationLevel.NO_COMPILATION:
            raise ValueError("No compilation level is set.")

        from torch._dynamo.backends.registry import list_backends
        torch_backends = list_backends(exclude_tags=tuple())
        if self.level in [
                CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE
        ]:
            if self.backend == "":
                return "eager"
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        # TODO: pass user-specified backend to piecewise compilation
        # merge with the config use_inductor
        assert self.level == CompilationLevel.PIECEWISE

        from vllm.compilation.backends import VllmBackend
        return VllmBackend(vllm_config)

    def init_with_cudagraph_sizes(self,
                                  cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""

        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            # de-duplicate the sizes provided by the config
            dedup_sizes = list(set(self.cudagraph_capture_sizes))
            if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
                logger.info(("cudagraph sizes specified by model runner"
                             " %s is overridden by config %s"),
                            cudagraph_capture_sizes, dedup_sizes)
            self.cudagraph_capture_sizes = dedup_sizes

        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", \
                    "Unrecognized size type in compile_sizes, " \
                    f"expect 'cudagraph_capture_sizes', got {x}"
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[
            0] if self.cudagraph_capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_capture_size + 1)
        ]
        for end, start in zip(self.cudagraph_capture_sizes,
                              self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
        self.bs_to_padded_graph_size[
            self.max_capture_size] = self.max_capture_size

    def set_splitting_ops_for_v1(self):
        # NOTE: this function needs to be called
        if self.splitting_ops and self.full_cuda_graph:
            raise ValueError("full_cuda_graph cannot be used together with "
                             "splitting_ops, as Full CUDA graph will override "
                             f"the splitting_ops: {self.splitting_ops}")

        if not self.splitting_ops:
            self.splitting_ops = [] if self.full_cuda_graph else [
                "vllm.unified_attention",
                "vllm.unified_attention_with_output",
                "vllm.mamba_mixer2",
            ]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = None  # type: ignore
    """Model configuration."""
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration."""
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = field(default_factory=LoadConfig)
    """Load configuration."""
    lora_config: Optional[LoRAConfig] = None
    """LoRA configuration."""
    speculative_config: Optional[SpeculativeConfig] = None
    """Speculative decoding configuration."""
    decoding_config: DecodingConfig = field(default_factory=DecodingConfig)
    """Decoding configuration."""
    observability_config: Optional[ObservabilityConfig] = None
    """Observability configuration."""
    quant_config: Optional[QuantizationConfig] = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, `-O<n>` can be used to directly specify the compilation
    level `n`: `-O3` is equivalent to `-O.level=3` (same as `-O='{"level":3}'`).
    Currently, -O <n> and -O=<n> are supported as well but this will likely be
    removed in favor of clearer -O<n> syntax in the future.

    NOTE: level 0 is the default level without any optimization. level 1 and 2
    are for internal testing only. level 3 is the recommended level for
    production, also default in V1.

    You can specify the full compilation config like so:
    `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    kv_transfer_config: Optional[KVTransferConfig] = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: Optional[KVEventsConfig] = None
    """The configurations for event publishing."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: Union[dict, SupportsHash] = field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the vLLM instance."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        from vllm import __version__
        vllm_factors.append(__version__)
        vllm_factors.append(envs.VLLM_USE_V1)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
            # LoRA creates static buffers based on max_num_batched_tokens.
            # The tensor sizes and strides get captured in the torch.compile
            # graph explicitly.
            vllm_factors.append(
                str(self.scheduler_config.max_num_batched_tokens))
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.decoding_config:
            vllm_factors.append(self.decoding_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.observability_config:
            vllm_factors.append(self.observability_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = hashlib.md5(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_capture_size
        return self.compilation_config.bs_to_padded_graph_size[batch_size]

    @staticmethod
    def _get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        """Get the quantization config."""
        from vllm.platforms import current_platform
        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import (
                get_quant_config)
            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}.")
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}")
            return quant_config
        return None

    @staticmethod
    def get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(copy.deepcopy(model_config),
                                                   load_config)

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: Optional[list[str]] = None,
    ) -> "VllmConfig":
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config

        return replace(self, model_config=model_config)

    def __post_init__(self):
        """Verify configs are valid & consistent with each other.
        """

        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config,
                                                       self.speculative_config,
                                                       self.device_config)
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(
                self.load_config)

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        from vllm.platforms import current_platform
        if self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.enable_async_tp:
            self.compilation_config.pass_config.enable_sequence_parallelism = \
                True
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append("+rms_norm")
        if envs.VLLM_USE_V1 and self.model_config is not None and \
            not self.model_config.enforce_eager:
            # By default, V1 uses piecewise CUDA graphs. If full_cuda_graph
            # is set to True, full CUDA graphs will be used.
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.level = CompilationLevel.PIECEWISE
            self.compilation_config.set_splitting_ops_for_v1()

        self._set_cudagraph_sizes()

        if self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION \
                and not envs.VLLM_USE_V1:
            logger.warning(
                "CPU offload is not supported with `torch.compile` in v0 yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if ((not envs.VLLM_USE_V1) and self.lora_config is not None
                and self.compilation_config.level
                != CompilationLevel.NO_COMPILATION):
            logger.warning(
                "LoRA for V0 is not supported with `torch.compile` yet. "
                "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if self.compilation_config.full_cuda_graph and \
            not self.model_config.disable_cascade_attn:
            logger.info("full_cuda_graph is not supported with "
                        "cascade attention. Disabling cascade attention.")
            self.model_config.disable_cascade_attn = True

        disable_chunked_prefill_reasons: list[str] = []

        if self.model_config and self.model_config.pooler_config:
            pooling_type = self.model_config.pooler_config.pooling_type
            if pooling_type is None or pooling_type.lower() != "last":
                disable_chunked_prefill_reasons.append(
                    "Only \"last\" pooling supports chunked "
                    "prefill and prefix caching; disabling both.")

        if disable_chunked_prefill_reasons:
            for reason in disable_chunked_prefill_reasons:
                logger.info(reason)
            self.scheduler_config.chunked_prefill_enabled = False
            self.scheduler_config.long_prefill_token_threshold = 0
            self.scheduler_config.max_num_batched_tokens = max(
                self.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.cache_config is not None:
                self.cache_config.enable_prefix_caching = False

        if (self.kv_events_config is not None
                and self.kv_events_config.enable_kv_cache_events
                and not self.cache_config.enable_prefix_caching):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled."
                "Use --enable-prefix-caching to enable.")
        if (self.kv_events_config is not None
                and self.kv_events_config.publisher != "null"
                and not self.kv_events_config.enable_kv_cache_events):
            logger.warning("KV cache events are disabled,"
                           "but the scheduler is configured to publish them."
                           "Modify KVEventsConfig.enable_kv_cache_events"
                           "to True to enable.")
        current_platform.check_and_update_config(self)

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        if (envs.VLLM_USE_V1
                and not self.scheduler_config.disable_hybrid_kv_cache_manager):
            # logger should only print warning message for hybrid models. As we
            # can't know whether the model is hybrid or not now, so we don't log
            # warning message here and will log it later.
            if not (current_platform.is_cuda() or current_platform.is_rocm()):
                # Hybrid KV cache manager is not supported on non-GPU platforms.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_transfer_config is not None:
                # Hybrid KV cache manager is not compatible with KV transfer.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_events_config is not None:
                # Hybrid KV cache manager is not compatible with KV events.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.model_config is not None and \
                self.model_config.attention_chunk_size is not None and \
                self.speculative_config is not None and \
                self.speculative_config.use_eagle():
                # Hybrid KV cache manager is not yet supported with chunked
                # local attention + eagle.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True

    def update_sizes_for_sequence_parallelism(self,
                                              possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled", removed_sizes,
                self.parallel_config.tensor_parallel_size)

        return [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_cudagraph_sizes(self):
        """
        cudagraph batchsize padding logic:

        `[1, 2, 4] + [8 * i for i in range(1, 1025)]` is a list of all possible
        batch sizes that cudagraph will capture.

        Depending on the engine's configuration of `max_num_seqs`, the
        candidate batch sizes to capture cudagraph will shrink to the subset
        which just cover the range of `[1, max_num_seqs]`. In the common case,
        `max_num_seqs` is 256, and the cudagraph batch sizes will be
        `[1, 2, 4, 8, 16, 24, 32, 40, ..., 256]`.

        However, if users specify the cudagraph capture sizes through
        compilation config, we will use the specified sizes instead.

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in descending order).

        During runtime, if batchsize is larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        no cudagraph will be used.
        If the batch size is no larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        we can quickly find the padded graph size for a given batch size by
        looking up `vllm_config.compilation_config.bs_to_padded_graph_size`.
        """

        # calculate the default `batch_size_capture_list`
        if not envs.VLLM_USE_V1:
            batch_size_capture_list = []
            if self.scheduler_config is not None and \
                self.model_config is not None and \
                    not self.model_config.enforce_eager:

                possible_sizes = [1, 2, 4] + [8 * i for i in range(1, 1025)]
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    possible_sizes = self.update_sizes_for_sequence_parallelism(
                        possible_sizes)

                # find the minimum size that is larger than max_num_seqs,
                # which then becomes the max_batchsize_to_capture
                larger_sizes = [
                    x for x in possible_sizes
                    if x >= self.scheduler_config.max_num_seqs
                ]
                if larger_sizes:
                    max_batchsize_to_capture = larger_sizes[0]
                else:
                    max_batchsize_to_capture = possible_sizes[-1]

                # filter out the sizes that are
                # larger than max_batchsize_to_capture
                batch_size_capture_list = [
                    size for size in possible_sizes
                    if size <= max_batchsize_to_capture
                ]
        else:
            batch_size_capture_list = []
            if self.model_config is not None and \
                not self.model_config.enforce_eager:
                cuda_graph_sizes = self.scheduler_config.cuda_graph_sizes
                if len(cuda_graph_sizes) == 1:
                    batch_size_capture_list = [1, 2, 4] + [
                        i for i in range(8, cuda_graph_sizes[0] + 1, 8)
                    ]
                elif len(cuda_graph_sizes) > 1:
                    batch_size_capture_list = sorted(cuda_graph_sizes)
                else:
                    raise TypeError(f"Invalid value for {cuda_graph_sizes=}.")
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    batch_size_capture_list = \
                        self.update_sizes_for_sequence_parallelism(batch_size_capture_list)
                max_num_tokens = self.scheduler_config.max_num_batched_tokens
                batch_size_capture_list = [
                    size for size in batch_size_capture_list
                    if size <= max_num_tokens
                ]

        self.compilation_config.init_with_cudagraph_sizes(
            batch_size_capture_list)

    def recalculate_max_model_len(self, max_model_len: int):
        # Can only be called in try_verify_and_update_config
        model_config = self.model_config
        max_model_len = model_config.get_and_verify_max_len(max_model_len)
        self.model_config.max_model_len = max_model_len
        self.scheduler_config.max_model_len = max_model_len

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig)
        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import (
                SequenceClassificationConfig)
            SequenceClassificationConfig.verify_and_update_config(self)

    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"override_neuron_config={self.model_config.override_neuron_config}, "  # noqa
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"decoding_config={self.decoding_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"num_scheduler_steps={self.scheduler_config.num_scheduler_steps}, "
            f"multi_step_stream_outputs={self.scheduler_config.multi_step_stream_outputs}, "  # noqa
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"chunked_prefill_enabled={self.scheduler_config.chunked_prefill_enabled}, "  # noqa
            f"use_async_output_proc={self.model_config.use_async_output_proc}, "
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}")


_current_vllm_config: Optional[VllmConfig] = None
_current_prefix: Optional[str] = None


@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig,
                            check_compile=False,
                            prefix: Optional[str] = None):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from vllm.compilation.counter import compilation_counter
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        logger.debug("enabled custom ops: %s",
                     vllm_config.compilation_config.enabled_custom_ops)
        logger.debug("disabled custom ops: %s",
                     vllm_config.compilation_config.disabled_custom_ops)
        if check_compile and \
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE \
            and compilation_counter.num_models_seen == num_models_seen:
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model)
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current vLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config


def get_current_model_prefix() -> str:
    """
    Get the prefix of the model that's currently being initialized.
    """
    assert _current_prefix is not None, \
        "Current model prefix is not set. "
    return _current_prefix


def contains_object_print(text):
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64 bit system).

    Args:
        text (str): The text to check

    Returns:
        result (bool): `True` if a match is found, `False` otherwise.
    """
    pattern = r'at 0x[a-fA-F0-9]{2,16}>'
    match = re.search(pattern, text)
    return match is not None


def assert_hashable(text):
    if not contains_object_print(text):
        return True
    raise AssertionError(
        f"vLLM tried to hash some configs that may have Python objects ids "
        f"in them. This is a bug, please file an issue. "
        f"Text being hashed: {text}")


T = TypeVar("T")


def get_layers_from_vllm_config(vllm_config: VllmConfig,
                                layer_type: type[T]) -> dict[str, T]:
    return {
        layer_name: layer
        for layer_name, layer in
        vllm_config.compilation_config.static_forward_context.items()
        if isinstance(layer, layer_type)
    }


@config
@dataclass
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""

    sample_rate: float = 16_000
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    max_audio_clip_s: int = 30
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected."""

    overlap_chunk_second: int = 1
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    min_energy_split_window_size: Optional[int] = 1600
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property
    def allow_audio_chunking(self) -> bool:
        return self.min_energy_split_window_size is not None


def update_config(config: DataclassInstanceT,
                  overrides: dict[str, Any]) -> DataclassInstanceT:
    processed_overrides = {}
    for field_name, value in overrides.items():
        assert hasattr(
            config, field_name), f"{type(config)} has no field `{field_name}`"
        current_value = getattr(config, field_name)
        if is_dataclass(current_value) and not is_dataclass(value):
            assert isinstance(value, dict), (
                f"Overrides to {type(config)}.{field_name} must be a dict"
                f"  or {type(current_value)}, but got {type(value)}")
            value = update_config(
                current_value,  # type: ignore[type-var]
                value)
        processed_overrides[field_name] = value
    return replace(config, **processed_overrides)
