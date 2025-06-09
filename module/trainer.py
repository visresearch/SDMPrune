

import importlib.metadata

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from transformers import (
    PreTrainedModel,
    Trainer,
)
from packaging import version
from transformers.trainer import logger
from transformers.trainer_utils import check_target_module_exists
from transformers.trainer_pt_utils import LayerWiseDummyOptimizer
from transformers.utils import (
    is_accelerate_available,
    is_bitsandbytes_available,
    is_galore_torch_available,
    strtobool,
    is_lomo_available,
    is_grokadamw_available,
    is_torchao_available,
    is_schedulefree_available,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.optimization import Adafactor
class PackedSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(model.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
        model = model.module if hasattr(model, 'module') else model
        loss_kwargs = {"num_items_in_batch":num_items_in_batch}
        # flash_attn don't need "num_items_in_batch"
        # if num_items_in_batch is not None:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch
        outputs = model(**inputs, **loss_kwargs, return_dict=True)
        original_loss = outputs.loss
        # data = {"loss":original_loss, "inputs":inputs, "num_items_in_batch":num_items_in_batch}
        # print(num_items_in_batch, original_loss)
        # torch.save(data, f"data{dist.get_rank()}.pth")
        # sys.exit(0)
        return (original_loss, outputs) if return_outputs else original_loss
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            # zhr
            # flash_attn don't need num_items_in_batch
            # loss = loss / num_items_in_batch
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):

            torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()
    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim in [
            OptimizerNames.ADAMW_BNB,
            OptimizerNames.ADAMW_8BIT,
            OptimizerNames.PAGED_ADAMW,
            OptimizerNames.PAGED_ADAMW_8BIT,
            OptimizerNames.ADEMAMIX,
            OptimizerNames.ADEMAMIX_8BIT,
            OptimizerNames.PAGED_ADEMAMIX,
            OptimizerNames.PAGED_ADEMAMIX_8BIT,
            OptimizerNames.LION,
            OptimizerNames.LION_8BIT,
            OptimizerNames.PAGED_LION,
            OptimizerNames.PAGED_LION_8BIT,
            OptimizerNames.RMSPROP_BNB,
            OptimizerNames.RMSPROP_8BIT,
            OptimizerNames.RMSPROP_32BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamW, Lion, RMSprop

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamW
                elif "lion" in args.optim:
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
                elif "rmsprop" in args.optim:
                    optimizer_cls = RMSprop
                    # Above we pass all `adam_kwargs` to the optimizer, here
                    # we only pass `optim_args` which can be passed by the user.
                    additional_optim_kwargs = optim_args
                elif "ademamix" in args.optim:
                    if is_bitsandbytes_available() and version.parse(
                        importlib.metadata.version("bitsandbytes")
                    ) < version.parse("0.44.0"):
                        raise ValueError(
                            "The AdEMAMix optimizer is not supported by your current version of `bitsandbytes`. "
                            "Please install `bitsandbytes` >= 0.44.0."
                        )

                    from bitsandbytes.optim import AdEMAMix

                    optimizer_cls = AdEMAMix
                    additional_optim_kwargs = {
                        "betas": (
                            float(optim_args.get("beta1", args.adam_beta1)),
                            float(optim_args.get("beta2", args.adam_beta2)),
                            float(optim_args.get("beta3", 0.9999)),
                        ),
                        "alpha": float(optim_args.get("alpha", 5.0)),
                        "eps": float(optim_args.get("eps", args.adam_epsilon)),
                    }

                    if "t_alpha" in optim_args:
                        additional_optim_kwargs["t_alpha"] = int(optim_args["t_alpha"])

                    if "t_beta3" in optim_args:
                        additional_optim_kwargs["t_beta3"] = int(optim_args["t_beta3"])

                bnb_kwargs = {"optim_bits": optim_bits}
                if "rmsprop" not in args.optim:
                    bnb_kwargs["is_paged"] = is_paged

                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but `bitsandbytes` is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from .anyprecisionAdamw import AnyPrecisionAdamW
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        elif args.optim in [
            OptimizerNames.GALORE_ADAMW,
            OptimizerNames.GALORE_ADAMW_8BIT,
            OptimizerNames.GALORE_ADAFACTOR,
            OptimizerNames.GALORE_ADAMW_LAYERWISE,
            OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE,
            OptimizerNames.GALORE_ADAFACTOR_LAYERWISE,
        ]:
            if not is_galore_torch_available():
                raise ImportError(
                    "You need to install `galore_torch` in order to use GaLore optimizers"
                    " install it with `pip install git+https://github.com/jiaweizzhao/GaLore`"
                )
            from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit

            is_layerwise = args.optim.lower().endswith("layerwise")
            if is_layerwise and args.parallel_mode == ParallelMode.DISTRIBUTED:
                raise NotImplementedError("Layer-wise GaLore does not support DDP at this time")

            optimizer_mapping = {
                OptimizerNames.GALORE_ADAMW: GaLoreAdamW,
                OptimizerNames.GALORE_ADAMW_8BIT: GaLoreAdamW8bit,
                OptimizerNames.GALORE_ADAFACTOR: GaLoreAdafactor,
                OptimizerNames.GALORE_ADAMW_LAYERWISE: GaLoreAdamW,
                OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE: GaLoreAdamW8bit,
                OptimizerNames.GALORE_ADAFACTOR_LAYERWISE: GaLoreAdafactor,
            }

            optimizer_cls = optimizer_mapping[args.optim]

            if args.optim_target_modules is None:
                raise ValueError(
                    "You need to define a `optim_target_modules` in order to properly use GaLore optimizers"
                )

            if not isinstance(args.optim_target_modules, (list, str)):
                raise ValueError(
                    f"`optim_target_modules` has to be a list of strings, a string corresponding to a regex, or a specific module or 'all-linear', you passed {args.optim_target_modules}"
                )

            if model is None:
                raise ValueError("You need to pass a model in order to correctly initialize a GaLore optimizer.")

            logger.warning(
                "Activated GaLoRE fine-tuning, depending on your model size and hardware, the training might take a while before starting. Please be patient !"
            )

            all_linear = (
                isinstance(args.optim_target_modules, str)
                and args.optim_target_modules.replace("_", "-") == "all-linear"
            )

            galore_params = []
            galore_params_names = []
            for module_name, module in model.named_modules():
                target_module_exists, is_regex = check_target_module_exists(
                    args.optim_target_modules, module_name, return_is_regex=True
                )

                if not isinstance(module, nn.Linear):
                    # Warn in case we match but it's not a linear layer
                    if target_module_exists and not is_regex:
                        logger.warning(
                            f"{module_name} has been matched but ignored as GaLore only supports linear layers. Please double check your `optim_target_modules`!"
                        )

                    continue

                if not target_module_exists and not all_linear:
                    continue

                galore_params.append(module.weight)
                galore_params_names.append(module_name + ".weight")

            if len(galore_params) == 0:
                raise ValueError(
                    f"None of the target modules were found! ({args.optim_target_modules}). Please make sure to pass a valid `target_modules`."
                )

            non_galore_params = [p for n, p in model.named_parameters() if n not in galore_params_names]

            galore_optim_kwargs = {
                "rank": int(optim_args.pop("rank", 128)),
                "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
                "scale": float(optim_args.pop("scale", 0.25)),
                "proj_type": optim_args.pop("proj_type", "std"),
            }

            # The default args are from the official repository: https://github.com/jiaweizzhao/GaLore
            param_groups = [
                {"params": non_galore_params},
                {"params": galore_params, **galore_optim_kwargs},
            ]

            if is_layerwise:
                # For layer-wise optimizers, the optimization step is done through post accumulation
                # gradient hooks. The trick is to first attach these hooks to the model parameters then
                # create a dummy optimizer that will perform no-ops in the Trainer.
                # See the original implementation or the nice implementation from @hiyouga
                # here: https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
                if args.gradient_accumulation_steps != 1:
                    raise ValueError("Layerwise GaLoRE optimizer do not support gradient accumulation !")

                optimizer_dict = {}
                for param in non_galore_params:
                    param_groups = [{"params": [param]}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)
                for param in galore_params:
                    param_groups = [{"params": [param], **galore_optim_kwargs}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)

                def optimizer_hook(param):
                    if param.grad is not None:
                        optimizer_dict[param].step()
                        optimizer_dict[param].zero_grad()

                for param in model.parameters():
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(optimizer_hook)

                optimizer_cls = LayerWiseDummyOptimizer
                optimizer_kwargs.update({"optimizer_dict": optimizer_dict})

            optimizer_kwargs.update({"params": param_groups})

            if args.optim == OptimizerNames.GALORE_ADAFACTOR:
                optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            if not is_lomo_available():
                raise ImportError(
                    "You need to install `lomo_optim` in order to use LOMO optimizers"
                    " install it with `pip install lomo-optim`"
                )
            if not is_accelerate_available("0.30.0"):
                raise ImportError("You need to have `accelerate>=0.30.0` to be able to use LOMO optimizers")

            if model is None:
                raise ValueError("You need to pass a `model` in order to correctly initialize a LOMO optimizer.")

            from lomo_optim import AdaLomo, Lomo

            if "ada" in args.optim:
                optimizer_cls = AdaLomo
            else:
                optimizer_cls = Lomo

            optimizer_kwargs.update({"model": model})
        elif args.optim == OptimizerNames.GROKADAMW:
            if not is_grokadamw_available():
                raise ValueError("Please install grokadamw with `pip install grokadamw`")

            from grokadamw import GrokAdamW

            optimizer_cls = GrokAdamW
            optimizer_kwargs.update(
                {
                    "alpha_init": float(optim_args.get("alpha_init", 0.98)),
                    "lamb": float(optim_args.get("lamb", 2.0)),
                    "gamma": float(optim_args.get("gamma", 0.1)),
                    "grokking_signal_decay_rate": float(optim_args.get("grokking_signal_decay_rate", 0.1)),
                    "gradient_clipping": float(optim_args.get("gradient_clipping", 1.0)),
                }
            )
        elif args.optim == OptimizerNames.ADAMW_TORCH_4BIT:
            if not is_torchao_available() or version.parse(importlib.metadata.version("torchao")) < version.parse(
                "0.4.0"
            ):
                raise ImportError(
                    "You need to have `torchao>=0.4.0` in order to use torch 4-bit optimizers."
                    "Install it with `pip install torchao` or follow the instructions here: https://github.com/pytorch/ao"
                )
            if version.parse(importlib.metadata.version("torch")) <= version.parse("2.4"):
                raise ImportError(
                    "You need to have `torch>2.4` in order to use torch 4-bit optimizers. "
                    "Install it with `pip install --upgrade torch` it is available on pipy. Otherwise, you need to install torch nightly."
                )
            from torchao.prototype.low_bit_optim import AdamW4bit

            optimizer_cls = AdamW4bit
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [
            OptimizerNames.SCHEDULE_FREE_ADAMW,
            OptimizerNames.SCHEDULE_FREE_SGD,
        ]:
            if not is_schedulefree_available():
                raise ImportError(
                    "You need to install `schedulefree` in order to use schedulefree optimizers"
                    " install it with `pip install schedulefree`"
                )
            if not is_accelerate_available("0.30.0"):
                raise ImportError("You need to have `accelerate>=0.30.0` to be able to use schedulefree optimizers")
            from schedulefree import AdamWScheduleFree, SGDScheduleFree

            additional_optim_kwargs = {}
            if args.optim == OptimizerNames.SCHEDULE_FREE_ADAMW:
                optimizer_cls = AdamWScheduleFree
                additional_optim_kwargs = adam_kwargs
            elif args.optim == OptimizerNames.SCHEDULE_FREE_SGD:
                optimizer_cls = SGDScheduleFree
            else:
                raise ValueError("Invalid schedulefree optimizer")
            additional_optim_kwargs["weight_decay"] = args.weight_decay
            additional_optim_kwargs["warmup_steps"] = args.warmup_steps
            additional_optim_kwargs.update(
                {
                    "weight_lr_power": float(optim_args.get("weight_lr_power", 2.0)),
                    "r": float(optim_args.get("r", 0.0)),
                }
            )
            optimizer_kwargs.update(additional_optim_kwargs)
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs