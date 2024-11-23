from tiny_dashboard import AbstractOnlineFeatureCentricDashboard
from nnterp.nnsight_utils import get_layer_output, get_layer
import torch as th


class CrosscoderOnlineFeatureDashboard(AbstractOnlineFeatureCentricDashboard):
    """
    Dashboard for analyzing features using a crosscoder model that combines base and instruct model activations.
    """

    def __init__(
        self,
        base_model,
        instruct_model,
        crosscoder,
        collect_layer: int,
        window_size: int = 50,
        crosscoder_device: str | None = None,
    ):
        """
        Args:
            base_model: Base language model
            instruct_model: Instruction-tuned model
            crosscoder: Model that combines base and instruct activations
            collect_layer: Layer to collect activations from
            window_size: Number of tokens to show before/after max activation
            crosscoder_device: Optional device to move crosscoder inputs to
        """
        super().__init__(instruct_model.tokenizer, instruct_model, window_size)
        self.base_model = base_model
        self.crosscoder = crosscoder
        self.crosscoder_device = crosscoder_device
        self.layer = collect_layer

    @th.no_grad()
    def get_feature_activation(
        self, text: str, feature_indices: tuple[int, ...]
    ) -> th.Tensor:
        """Get the activation values for given features by combining base and instruct model activations"""
        with self.model.trace(text):  # self.model is the instruct_model from parent
            instruct_activations = get_layer_output(self.model, self.layer)[0].save()
            get_layer(self.model, self.layer).output.stop()

        with self.base_model.trace(text):
            base_activations = get_layer_output(self.base_model, self.layer)[0].save()
            get_layer(self.base_model, self.layer).output.stop()

        if self.crosscoder_device is not None:
            base_activations = base_activations.to(self.crosscoder_device)
            instruct_activations = instruct_activations.to(self.crosscoder_device)

        cc_input = th.stack([base_activations, instruct_activations], dim=1).float()
        features_acts = self.crosscoder.get_activations(
            cc_input, select_features=list(feature_indices)
        )
        return features_acts
