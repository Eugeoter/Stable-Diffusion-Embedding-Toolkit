from __future__ import annotations
import os
import safetensors.torch
import torch

EMBEDDINGS_DIR = "./embeddings/"  # Path to embeddings


def load_model(model_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path, device=device)
    elif model_path.endswith(".pt"):
        state_dict = torch.load(model_path, map_location=device)
    else:
        raise TypeError(f"Unknown model type `{os.path.splitext(model_path)[1]}`.")

    return state_dict


def merge_models(
    state_dict_A: dict[str, torch.Tensor],
    state_dict_B: dict[str, torch.Tensor],
    alpha: float
) -> dict[str, torch.Tensor]:
    r"""
    Merge two model state dicts. Computed as :math:`\alpha \cdot A + (1 - \alpha) \cdot B`.
    """
    dim_A = tuple(state_dict_A['clip_g'].shape)[0]
    dim_B = tuple(state_dict_B['clip_g'].shape)[0]

    if dim_B > dim_A:
        return merge_models(state_dict_B, state_dict_A, 1 - alpha)

    merged_model = {}
    for key, weight_A in state_dict_A.items():
        weight_B = state_dict_B[key]

        dim_diff = dim_A - dim_B
        if dim_diff > 0:
            padding = torch.zeros((dim_diff, tuple(weight_A.shape)[1]))
            weight_B = torch.cat((weight_B, padding), dim=0)

        merged_weight = (1 - alpha) * weight_A + alpha * weight_B
        merged_model[key] = merged_weight

    return merged_model


def merge_embeddings(
    model_path_A: str,
    model_path_B: str,
    alpha: float,
    output_filename: str = None,
    save_as_safetensors: bool = True,
    allow_overwrite_output: bool = False
) -> None:
    r"""
    Merge two embeddings. Computed as :math:`\alpha \cdot A + (1 - \alpha) \cdot B`.
    """
    output_filename = os.path.splitext(output_filename)[0] + ('.safetensors' if save_as_safetensors else ".pt")

    if not allow_overwrite_output and os.path.exists(output_filename):
        raise FileExistsError(f"Output file `{os.path.abspath(output_filename)}` already exists.")

    state_dict_A = load_model(model_path_A, device="cpu")
    state_dict_B = load_model(model_path_B, device="cpu")

    merged_state_dict = merge_models(state_dict_A, state_dict_B, alpha)

    if save_as_safetensors:
        safetensors.torch.save_file(merged_state_dict, output_filename)
    else:
        torch.save(merged_state_dict, output_filename)


def get_model_type(state_dict: dict[str, torch.Tensor]):
    """
    Get the type of model from the state dict. Either 'XL' or '1.5'.
    """
    if 'clip_g' in state_dict.keys() and 'clip_l' in state_dict.keys():
        return 'XL'
    elif 'string_to_token' in state_dict.keys() and 'string_to_param' in state_dict.keys():
        return '1.5'


def get_model_dim(state_dict: dict[str, torch.Tensor]):
    """
    Get the dimension of the model from the state dict.
    """
    model_type = get_model_type(state_dict)
    if model_type == 'XL':
        return tuple(state_dict['clip_l'].shape)[0]
    elif model_type == '1.5':
        return tuple(state_dict['string_to_param']['*'].shape)[0]


def merge_embedding_list(
    model_paths: list[str],
    boosts: list[float],
    as_type: str,
    output_filename: str = None,
    save_as_safetensors: bool = True,
    allow_overwrite_output: bool = False
) -> None:
    r"""
    Merge a list of embeddings. Computed as :math:`\sum_i \alpha_i \cdot A_i`.
    :param model_paths: List of model paths.
    :param boosts: List of boosts.
    :param as_type: Type of model to save as. Either 'XL' or '1.5'.
    :param output_filename: Output filename. The extension will be changed to '.safetensors' if `save_as_safetensors` is `True`.
    :param save_as_safetensors: Whether to save as a `.safetensors` file.
    :param allow_overwrite_output: Whether to allow overwriting the output file.
    """
    if as_type == '1.5' and save_as_safetensors:
        save_as_safetensors = False

    output_filename = os.path.splitext(output_filename)[0] + ('.safetensors' if save_as_safetensors else ".pt")

    if not allow_overwrite_output and os.path.exists(output_filename):
        raise FileExistsError(f"Output file `{os.path.abspath(output_filename)}` already exists.")

    state_dicts = [load_model(model_path) for model_path in model_paths]

    merged_state_dict = merge_model_list(state_dicts, boosts, as_type)

    if as_type == '1.5':
        merged_state_dict['string_to_param']['name'] = os.path.basename(os.path.splitext(output_filename)[0])

    if save_as_safetensors:
        safetensors.torch.save_file(merged_state_dict, output_filename)
    else:
        torch.save(merged_state_dict, output_filename)


def merge_model_list(
    state_dicts: dict[str, torch.Tensor],
    boosts: list[float],
    as_type: str
):
    r"""
    Merge a list of models. Computed as :math:`\sum_i \alpha_i \cdot A_i`.
    """
    state_dicts = [convert_model_as(sd, as_type) for sd in state_dicts]

    dims = [get_model_dim(sd) for sd in state_dicts]
    max_dim = max(dims)

    if as_type == 'XL':
        output_state_dict = {
            'clip_g': 0,
            'clip_l': 0
        }

    elif as_type == '1.5':
        output_state_dict = {
            'string_to_token': {'*': 265},
            'string_to_param': {
                '*': 0,
                'name': None,
                'step': None,
                'sd_checkpoint': None,
                'sd_checkpoint_name': None
            }
        }

    for i, state_dict in enumerate(state_dicts):
        if as_type == 'XL':
            for key, weight in state_dict.items():
                dim_diff = max_dim - dims[i]
                if dim_diff > 0:
                    padding = torch.zeros((dim_diff, tuple(weight.shape)[1]))
                    weight = torch.cat((weight, padding), dim=0)

                output_state_dict[key] += boosts[i] * weight

        elif as_type == '1.5':
            weight = state_dict['string_to_param']['*']

            dim_diff = max_dim - dims[i]
            if dim_diff > 0:
                padding = torch.zeros((dim_diff, tuple(weight.shape)[1]))
                weight = torch.cat((weight, padding), dim=0)

            output_state_dict['string_to_param']['*'] += boosts[i] * weight

    return output_state_dict


def boost_model(state_dict: dict[str, torch.Tensor], boost: float) -> dict[str, torch.Tensor]:
    r"""
    Boost the weight of a model by a factor of :math:`\alpha`. Computed as :math:`\alpha \cdot A`.
    """
    if boost < 0:
        raise ValueError(f"The value of `boost` should be larger than 0.")

    boosted_state_dict = {}

    for key, weight in state_dict.items():
        boosted_weight = weight * boost
        boosted_state_dict[key] = boosted_weight

    return boosted_state_dict


def boost_embedding(
    model_path: str,
    boost: float,
    output_filename: str = None,
    save_as_safetensors: bool = True,
    allow_overwrite_output: bool = False
) -> None:
    r"""
    Boost the weight of an embedding by a factor of :math:`\alpha`. Computed as :math:`\alpha \cdot A`.
    """
    output_filename = os.path.splitext(output_filename)[0] + ('.safetensors' if save_as_safetensors else '.pt')

    if not allow_overwrite_output and os.path.exists(output_filename):
        raise FileExistsError(f"Output file `{os.path.abspath(output_filename)}` already exists.")

    state_dict = load_model(model_path, device="cpu")

    boosted_state_dict = boost_model(state_dict, boost)

    if save_as_safetensors:
        safetensors.torch.save_file(boosted_state_dict, output_filename)
    else:
        torch.save(boosted_state_dict, output_filename)


def convert_model_as(state_dict: dict[str, torch.Tensor], as_type: str) -> dict[str, torch.Tensor]:
    r"""
    Convert a model to a different type.
    :param state_dict: The state dict of the model to convert.
    :param as_type: The type to convert to.
    :return: The state dict of the converted model.
    """
    if as_type == 'XL':
        if get_model_type(state_dict) == 'XL':
            return state_dict

        l_weight = state_dict['string_to_param']['*']
        dim = tuple(l_weight.shape)[0]

        output_state_dict = {
            'clip_g': torch.zeros((dim, 1280)),
            'clip_l': l_weight
        }

    elif as_type == '1.5':
        if get_model_type(state_dict) == '1.5':
            return state_dict

        l_weight = state_dict['clip_l']

        output_state_dict = {
            'string_to_token': {'*': 265},
            'string_to_param': {
                '*': l_weight,
                'name': None,
                'step': None,
                'sd_checkpoint': None,
                'sd_checkpoint_name': None
            }
        }

    return output_state_dict


def convert_embedding_as(
    model_path: str,
    as_type: str,
    output_filename: str,
    save_as_safetensors: bool = True,
    allow_overwrite_output: bool = False
) -> None:
    r"""
    Convert an embedding to a different type.
    :param model_path: The path to the model to convert.
    :param as_type: The type to convert to.
    :param output_filename: The path to save the converted model to.
    """
    output_filename = os.path.splitext(output_filename)[0] + ('.safetensors' if save_as_safetensors else '.pt')

    if not allow_overwrite_output and os.path.exists(output_filename):
        raise FileExistsError(f"Output file `{os.path.abspath(output_filename)}` already exists.")

    state_dict = load_model(model_path, device="cpu")

    output_state_dict = convert_model_as(state_dict, as_type)

    if as_type == '1.5':
        output_state_dict['string_to_param']['name'] = os.path.basename(os.path.splitext(output_filename)[0])

    if save_as_safetensors:
        safetensors.torch.save_file(output_state_dict, output_filename)
    else:
        torch.save(output_state_dict, output_filename)


if __name__ == "__main__":
    # Example usage of emb model merging
    model_path_A = "./models/embeddings/aidxl_20ep.safetensors"  # Path to model A
    model_path_B = "./models/embeddings/aid210_XL.safetensors"  # Path to model B

    alpha = 0.5  # Output = (1 - alpha) * A + alpha * B

    output_model_path = "./models/embeddings/000.safetensors"  # Path to output model

    print(
        f"Start merging",
        f"  Embedding A: {os.path.basename(model_path_A)}",
        f"  Embedding B: {os.path.basename(model_path_B)}",
        f"  Alpha: {alpha}",
        sep='\n'
    )

    merge_embeddings(
        model_path_A,
        model_path_B,
        alpha=alpha,
        output_filename=output_model_path,
        save_as_safetensors=True,
        allow_overwrite_output=False
    )

    print(
        f"Merged!",
        f"Output file at: {os.path.abspath(output_model_path)}"
    )
