def compute_structural_mapping(human_features, model):
    """
    Computes Structural Mapping (SM) using a multimodal contrastive encoder.
    """
    return model.encode(human_features)
