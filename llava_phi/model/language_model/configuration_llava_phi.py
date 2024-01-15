import os
from typing import Union
from transformers import PretrainedConfig, PhiConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlavaPhiVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        mm_vision_select_feature (`str`, *optional*, defaults to `"patch"`):
            The feature to select from the vision encoder output. Can be one of `"patch"` or `"cls_patch"`.
        mm_vision_select_layer (`int`, *optional*, defaults to `-2`):
            The layer to select from the vision encoder output.

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_phi_clip_vision_model"

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            mm_vision_select_feature="patch",
            mm_vision_select_layer=-2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_phi-phi":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ProjectorConfig(PretrainedConfig):
    model_type = "llava_phi_projector"

    def __init__(
            self,
            mm_projector_type="linear",
            mm_hidden_size=768,
            hidden_size=2560,
            **kwargs
    ):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_phi-phi":
            config_dict = config_dict["projector_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


DEFAULT_VISUAL_CONFIG = {
    "vision_tower": LlavaPhiVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict()
}


class LlavaPhiConfig(PhiConfig):
    model_type = "llava_phi"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)


if __name__ == "__main__":
    print(LlavaPhiVisionConfig())
