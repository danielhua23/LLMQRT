import os
import gc
import warnings
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict

from runtime.nn_models.modules.linear import ( # 具体调gemm还是gemv还是ipex还是marlin，是由量化过程中写进config.json，位于autoAWQ/models/base.py#L215
    get_concrete_linear_module,
)
from runtime.utils.common_utils import (
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    try_import,
)
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoProcessor,
    BaseImageProcessor,
    ProcessorMixin,
    PreTrainedTokenizer,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from runtime.core.config import QuantConfig # both need in quantizer and runtime

TRANSFORMERS_AUTO_MAPPING_DICT = {
    "llama": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "qwen2_vl": "AutoModelForVision2Seq",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
}

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BaseModelForCausalLM(nn.Module):
    def __init__(
        self,
        model: Annotated[PreTrainedModel, Doc("The pretrained or quantized model.")],
        model_type: Annotated[str, Doc("The model type, found in config.json.")],
        is_quantized: Annotated[
            bool, Doc("Indicates if the current model is quantized.")
        ],
        config: Annotated[PretrainedConfig, Doc("The config of the model.")],
        quant_config: Annotated[
            QuantConfig, Doc("The quantization config of the model.")
        ],
        processor: Annotated[
            BaseImageProcessor, Doc("An optional processor, e.g. for vision models.")
        ],
    ):
        """The base model for all models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config
        self.processor: ProcessorMixin = processor

    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @torch.no_grad()
    def pack(self):
        """
        A utility function for the following scenario. Note that save_quantized will
        overwrite existing weights if you use the same quant_path.

        Example:

        ```python
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            export_compatible=True
        )
        model.save_quantized(...)  # produces GGUF/other compat weights
        model.pack(...) # makes the model CUDA compat
        model.save_quantized(...)  # produces CUDA compat weights
        ```
        """
        self.quantizer.pack()

    @staticmethod
    def fuse_layers(model):
        pass

    @classmethod
    def from_quantized(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        model_filename: Annotated[
            str, Doc("Load a specific model's filename by specifying this argument.")
        ] = "",
        max_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence cached sequence length of the model. Larger values may increase loading time and memory usage."
            ),
        ] = None,
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "The dtype to load the model as. May not work with other values than float16."
            ),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc(
                "Useful for Huggingface repositories that have not been integrated into transformers yet."
            ),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        fuse_layers: Annotated[
            bool,
            Doc(
                "Whether to use fused/optimized combination of layers for increased speed."
            ),
        ] = True,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = "auto",
        max_memory: Annotated[
            Dict[Union[int, str], Union[int, str]],
            Doc(
                'A dictionary device identifier to maximum memory which will be passed onto the model loading method from transformers. For example：{0: "4GB",1: "10GB"'
            ),
        ] = None,
        offload_folder: Annotated[
            str,
            Doc("The folder ot offload the model to."),
        ] = None,
        download_kwargs: Annotated[
            Dict,
            Doc("Used for configure download model"),
        ] = None,
        **config_kwargs: Annotated[
            Dict,
            Doc(
                "Additional kwargs that are passed to the config during initialization."
            ),
        ],
    ):
        """A method for initialization of a quantized model, usually in INT4. 
        即输入量化weight路径,返回含qweight和qlinear的quantized model"""
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            model_filename,
            safetensors,
            trust_remote_code,
            max_seq_len=max_seq_len,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name) # autoModelForCausalLM

        # [STEP 3] Load model(empty model only strutcture, no true weight)而且是标准的transformer qwen2，不是AWQlinear版本
        # 基于你提供的配置（config）和数据类型（torch_dtype）进行初始化,根据config里写的，返回Qwen2ForCausalLM的实例
        with init_empty_weights():
            model = target_cls.from_config( # 这个函数挺牛逼，有了这个之后，不需要以前那样的modeling_qwen3.py了
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        # import pdb;pdb.set_trace()
        # Prepare QuantizedLinear layers, replace nn.Linear, 仅仅构建了q linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            dtype=torch_dtype
        )

        model.tie_weights() # input和output embedding共享一个table
        # import pdb;pdb.set_trace()
        # 加载权重并应用量化 设备分发与内存优化, 经过了这个之后，weight全都到cuda7去了
        # ！！！！所以为了避免这种情况，我们设置CUDA VISIBLE DEVICES为0，强制都到0去
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map="auto",
            max_memory=max_memory,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )
        # 此时经过load_checkpoint_and_dispatch后的model变成了加载了awq quantizer信息的model了，scale也为fp16了（因为指定了torch dtype的原因）
        # import pdb;pdb.set_trace()
        # Dispath to devices
        if fuse_layers:
            # if llm_quant_runtime is None:
            #     warnings.warn("Skipping fusing modules because AWQ extension is not installed." + msg)
            # else:
            # import pdb;pdb.set_trace()
            self.fuse_layers(model) # 调用qwen3.py里面的fuse layers适配csrc kernels

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
            processor=None,
        )

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            if download_kwargs is None:
                download_kwargs = {}

            if "ignore_patterns" in download_kwargs:
                download_kwargs_ignore_patterns = download_kwargs.pop("ignore_patterns")

                if isinstance(download_kwargs_ignore_patterns, str):
                    ignore_patterns.append(download_kwargs_ignore_patterns)
                elif isinstance(download_kwargs_ignore_patterns, list):
                    ignore_patterns.extend(download_kwargs_ignore_patterns)

            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns, **download_kwargs
            )

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else: # 始终走这条路
            model_weights_path = model_path

        # [STEP 2] 加载model_path中的config json，读取出quant config，包括quant method，scale，zp等（这些是quantizer的时候写入的
        # TODO: Create BaseAWQConfig class
        quant_config = QuantConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config
    
    # # inplace load
    def _load_quantized_modules(
        self, model, quant_config, dtype=torch.float16
    ):
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            # layer为原始的fp16版本未量化版本的linear
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # 根据不同的量化方法dispatch到不同qliner gemm
            q_linear_module = get_concrete_linear_module(quant_config.quant_method) # WQLinear_GEMM
            # Replace nn.Linear with WQLinear，虽然模型里面本身就带着有，但是那不是我们自己写的triton kernel，所以这里还是要把runtime里的kernel替换过来
            for name, module in named_linears.items():
                # TO BE CHANGED
                q_linear = q_linear_module.from_linear( # q_linear_module必须是确定好的特定量化方法的linear class
                    # TODO: 需要quant config新增per tensor
                    module, quant_config.w_bit, quant_config.q_group_size, True, dtype=dtype, per_tensor=quant_config.per_tensor 
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

                torch.cuda.empty_cache()
            gc.collect()

    # called in _loaded_quantized_modules
    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer) # 返回一个空dict，目前autoAWQ中暂未看到scale_dict["is_scalable"]为true的情况

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)
