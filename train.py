#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path
from typing import Any, Callable, Final, Optional

import numpy as np
import torch
from python_tools.generic import namespace_as_string
from python_tools.ml import metrics, neural
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.default.neural_models import MLPModel
from python_tools.ml.default.transformations import (
    DefaultTransformations,
    revert_transform,
    set_transform,
)
from python_tools.ml.evaluator import evaluator
from python_tools.typing import DataLoaderT, TransformDict

from dataloader import get_partitions


class MLP_parallel(neural.LossModule):
    view_starts: Final[torch.Tensor]

    def __init__(
        self,
        *,
        input_size: int = -1,
        input_sizes: tuple[int, ...],
        interactions: bool,
        **kwargs,
    ) -> None:
        super().__init__(loss_function=kwargs["loss_function"])
        input_sizes = tuple(x for x in input_sizes if x)
        assert input_size == sum(input_sizes)
        assert len(input_sizes) <= 3

        self.view_starts = torch.cumsum(torch.LongTensor([0, *input_sizes]), dim=0)

        # before TFN
        layer_sizes: tuple[int, ...] = kwargs.pop("layer_sizes")
        output_size: int = kwargs.pop("output_size")
        final_activation: dict[str, Any] = kwargs.pop("final_activation")
        self.models = torch.nn.ModuleList(
            [
                neural.MLP(
                    input_size=size,
                    output_size=layer_sizes[-1] if layer_sizes else -1,
                    layer_sizes=layer_sizes[:-1] if layer_sizes else layer_sizes,
                    final_activation=kwargs["activation"],
                    **kwargs,
                )
                for size in input_sizes
            ]
        )

        # determine input size for final layer
        if layer_sizes:
            input_sizes = tuple([layer_sizes[-1]] * len(input_sizes))
        self.mult = neural.InteractionModel(input_sizes=input_sizes, append_one=True)
        products = self.mult(
            torch.cat(
                [
                    torch.ones(1, size) * value
                    for size, value in zip(input_sizes, [2, 3, 5])
                ],
                dim=1,
            ),
            meta={},
        )[0][0]
        self.product_indices = {
            "A": products == 2,
            "B": products == 3,
            "C": products == 5,
            "AB": products == 2 * 3,
            "AC": products == 2 * 5,
            "BC": products == 3 * 5,
            "ABC": products == 2 * 3 * 5,
        }
        self.product_indices = {
            key: value for key, value in self.product_indices.items() if value.sum() > 0
        }
        if interactions:
            assert products.shape[0] == sum(
                x.sum() for x in self.product_indices.values()
            )
        else:
            self.product_indices = {
                key: value
                for key, value in self.product_indices.items()
                if len(key) == 1
            }

        # after TFN
        self.linear = torch.nn.ModuleList(
            [
                neural.MLP(
                    input_size=indices.sum().item(),
                    output_size=output_size,
                    layer_sizes=(),
                    final_activation=final_activation,
                    **kwargs,
                )
                for indices in self.product_indices.values()
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # get unimodal embeddings
        ys = []
        for i, model in enumerate(self.models):
            start = self.view_starts[i]
            y_hat, meta = model(
                x[:, start : self.view_starts[i + 1]], meta, y=y, dataset=dataset
            )
            ys.append(y_hat)

        # reserve half the embedding for E-HGR loss
        if ys[0].shape == ys[-1].shape:
            meta["meta_loss_hgr_embeddings"] = torch.stack(ys, dim=1)

        # outer product
        products, meta = self.mult(
            torch.cat(ys, dim=1), meta=meta, dataset=dataset, y=y
        )
        ys = []
        for linear, (modalities, indices) in zip(
            self.linear, self.product_indices.items()
        ):
            y_hat, meta = linear(products[:, indices], meta=meta, dataset=dataset, y=y)
            ys.append(y_hat)
            meta[f"meta_loss_{modalities}"] = y_hat

        if "meta_loss_hgr_embeddings" not in meta:
            # only used when self.models has no learnable laryers
            meta["meta_loss_hgr_embeddings"] = torch.stack(
                ys[: len(self.models)], dim=1
            )
        meta["meta_loss_hgr_embeddings"] = meta["meta_loss_hgr_embeddings"][
            :, :, meta["meta_loss_hgr_embeddings"].shape[-1] // 2 :
        ]

        meta["meta_embedding"] = torch.stack(ys, dim=1)
        return meta["meta_embedding"].sum(dim=1), meta


def mask_shared(
    modalities: int, outputs_per_modality: int, final_outputs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cov_shape = modalities * outputs_per_modality
    index = torch.arange(outputs_per_modality)
    shared = torch.ones(cov_shape, cov_shape) != 1
    for imodality in range(modalities):
        imodality_index = index + imodality * outputs_per_modality
        for jmodality in range(imodality + 1, modalities):
            jmodality_index = index + jmodality * outputs_per_modality
            # first part is unique
            # pairwise shared
            if imodality == 0 and jmodality == 1:
                iblock = slice(final_outputs, final_outputs * 2)
                jblock = iblock
            elif imodality == 0 and jmodality == 2:
                iblock = slice(final_outputs * 2, final_outputs * 3)
                jblock = slice(final_outputs, final_outputs * 2)
            elif imodality == 1 and jmodality == 2:
                iblock = slice(final_outputs * 2, final_outputs * 3)
                jblock = iblock
            else:
                assert False
            shared[imodality_index[iblock], jmodality_index[jblock]] = True
    mask = (
        torch.arange(cov_shape)[:, None] + torch.arange(cov_shape)[None, :]
    ) % final_outputs
    mask = mask == mask.diag()
    same_modality = torch.arange(cov_shape) // outputs_per_modality
    same_modality = same_modality[:, None] == same_modality[None, :]
    same_modality[torch.nonzero(shared).reshape(-1)] = False
    return (shared & mask).triu(1), ((~shared) & mask & same_modality).triu(1)


def sum_within_modality(modalities: int, y_hat: torch.Tensor) -> torch.Tensor:
    return sum(y_hat.split(y_hat.shape[1] // modalities, dim=1))


class MLP_shared_unique(neural.LossModule):
    factors: Final[str]
    lambda_factor: Final[float]

    def __init__(
        self,
        *,
        factors: str = "",
        lambda_factor: float = 0.0,
        interactions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(loss_function=kwargs["loss_function"])

        kwargs["input_sizes"] = tuple(x for x in kwargs["input_sizes"] if x > 0)
        final_outputs = kwargs["output_size"]
        kwargs["output_size"] *= len(kwargs["input_sizes"])
        self.parallel = MLP_parallel(interactions=interactions, **kwargs)

        self.factors = factors
        self.lambda_factor = lambda_factor

        self.dep, self.indep = mask_shared(
            len(kwargs["input_sizes"]), kwargs["output_size"], final_outputs
        )

        self.jit_me = False

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_hat, meta = self.parallel.forward(x, meta, y=y, dataset=dataset)

        # combine unique+shared
        y_hat = sum_within_modality(len(self.parallel.models), y_hat)
        return y_hat, meta

    @torch.jit.ignore
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = meta.pop("meta_loss_hgr_embeddings")
        if self.factors.startswith("smurf") and len(self.parallel.models) < len(
            self.parallel.linear
        ):
            # MRO
            combined_y = None
            base = 0.0
            for imodal in range(len(self.parallel.models)):
                keys = [
                    f"meta_loss_{key}"
                    for key in self.parallel.product_indices
                    if len(key) == imodal + 1
                ]

                new_y = sum_within_modality(
                    len(self.parallel.models),
                    torch.stack([meta[key] for key in keys], dim=1).sum(dim=1),
                )
                if combined_y is None:
                    combined_y = new_y
                else:
                    combined_y = combined_y.detach() + new_y
                base = base + super().loss(combined_y, ground_truth, meta)
        else:
            base = super().loss(scores, ground_truth, meta)
        if self.lambda_factor == 0.0:
            return base
        corr_loss = 0.0

        if self.factors.startswith("ehgr"):
            # HGR in embedding
            embeddings = embeddings - embeddings.mean(dim=0, keepdims=True)
            hgrs = [
                (embeddings[:, i] * embeddings[:, j]).sum(dim=1).mean()
                - embeddings[:, i].T.cov().mm(embeddings[:, j].T.cov()).trace() / 2
                for i in range(embeddings.shape[1])
                for j in range(i + 1, embeddings.shape[1])
            ]
            corr_loss = -sum(hgrs)
        else:
            # SMUR'ed MRO
            # repeat SMURF for each interaction group with multiple modalities
            assert self.factors.startswith("smurf"), self.factors

            # self.dep and self.indep are the same for unimodal and bimodal for
            # three modalities (needs changes for more than three modalities)
            shared_var = torch.nonzero(self.dep).reshape(-1)

            for imodal in range(len(self.parallel.models)):
                keys = [
                    f"meta_loss_{key}"
                    for key in self.parallel.product_indices
                    if len(key) == imodal + 1
                ]
                if len(keys) < 2:
                    break

                cov = torch.cat([meta[key] for key in keys], dim=1).T.cov()
                corr_loss = (
                    corr_loss
                    # uncorrelated within a modality
                    + cov[self.indep].abs().mean()
                    # pairwise correlated across modalities
                    - cov[self.dep].mean()
                    # but do not only inflate variance
                    + cov[shared_var, shared_var].view(-1, 2).prod(dim=1).mean() / 2
                )
        return base + corr_loss * self.lambda_factor


def train(
    partitions: dict[int, dict[str, DataLoader]], folder: Path, args: argparse.Namespace
) -> None:
    # regression
    metric = "pearson_r_proportional"
    metric_max = True
    metric_fun = metrics.interval_metrics
    params = {"interval": True}
    # classification
    if args.dimension.startswith(("tpot", "vreed")):
        metric = "accuracy_proportional"
        metric_max = True
        metric_fun = metrics.nominal_metrics
        params = {"nominal": True}

    clustering = not args.dimension.startswith(("mos", "vreed"))
    if not clustering:
        metric = metric.removesuffix("_proportional")

    grid_search = {
        "epochs": [10_000],
        "early_stop": [300],
        "lr": [0.001, 0.0001, 0.00001, 0.005, 0.01, 0.05],
        "dropout": [0.0],
        "layers": [0],
        "layer_sizes": [(5,), (10,), (20, 10), (100, 20, 10), (100, 100, 10)],
        "activation": [{"name": "ReLU"}],
        "attenuation": [""],
        "sample_weight": [False],
        "final_activation": [{"name": "linear"}],
        "minmax": [False],
        "weight_decay": [1e-4, 1e-3, 1e-2, 0.0],
        "loss_function": ["MSELoss"],
        "model_class": [MLP_shared_unique],
        "factors": [args.model],
        "interactions": [args.interactions == 1],
        # early stopping
        "metric": ["optimization"],
        "metric_max": [False],
    }

    if args.dimension.startswith(("tpot", "vreed")):
        grid_search["loss_function"] = ["CrossEntropyLoss"]
        grid_search["class_weight"] = [False]
    elif args.dimension.endswith("toy"):
        # one linear layer
        grid_search["layer_sizes"] = [()]
        grid_search["layers"] = [-1]
        grid_search["weight_decay"] = grid_search["weight_decay"][-1:]
        grid_search["loss_function"] = grid_search["loss_function"][:1]
        grid_search["activation"] = [{"name": "linear"}]
        grid_search["metric"] = ["epoch"]
        grid_search["epochs"] = [1_000]
        grid_search["metric_max"] = [True]

    model = MLPModel(device="cuda", **params)

    if "base" not in args.model:
        grid_search["lambda_factor"] = [1.0, 0.75, 0.5, 0.25, 0.1]
        model.forward_names = (*model.forward_names, "lambda_factor")

    # remove modalities
    x_names = partitions[0]["training"].properties["x_names"].copy()
    if "a+l" in args.modalities:
        # mask language as acoustic for UMEME
        x_names = np.array([x.replace("language_", "acoustic_") for x in x_names])
    keep = np.ones(len(x_names), dtype=bool)
    for iname, name in enumerate(x_names):
        if (
            (name.startswith("acoustic") and "a" not in args.modalities)
            or (name.startswith("language") and "l" not in args.modalities)
            or (name.startswith("vision") and "v" not in args.modalities)
            or (name.startswith("ecg") and "h" not in args.modalities)
            or (name.startswith("eda") and "s" not in args.modalities)
            or (name.startswith("mocap") and "m" not in args.modalities)
        ):
            keep[iname] = False
    for fold in partitions.values():
        for partition in fold.values():
            partition.properties["x_names"] = x_names[keep].copy()
            for batch in partition:
                batch["x"][0] = batch["x"][0][:, keep]
                if "m" in args.modalities:
                    # remove samples with NaNs
                    samples_keep = np.isfinite(batch["x"][0]).all(axis=1)
                    for key in batch:
                        batch[key][0] = batch[key][0][samples_keep]
    x_names = x_names[keep]

    # add feature names
    new_names = ["input_sizes", "factors", "interactions"]
    grid_search["input_sizes"] = [  # sorted a-z!
        tuple(
            tuple(x for x in x_names if x.startswith(prefix))
            for prefix in sorted(
                ("acoustic", "ecg", "eda", "language", "mocap", "vision")
            )
        )
    ]
    model.forward_names = (*model.forward_names, *new_names)

    model.parameters.update(grid_search)
    models, parameters, model_transform = model.get_models()

    apply_transformation = partial(
        combine_transformations, model_transform=model_transform
    )

    transform = DefaultTransformations(**params)
    transforms = tuple([{"feature_selection": args.fs} for _ in range(len(partitions))])

    print(folder, len(parameters))
    for key, value in model.parameters.items():
        if len(value) > 1:
            print(len(value), key, value)

    metric_fun = partial(
        metric_fun,
        which=(metric,),
        clustering=clustering,
        names=tuple(partitions[0]["training"].properties["y_names"].tolist()),
    )

    evaluator(
        models=models,
        partitions=partitions,
        parameters=parameters,
        folder=folder,
        metric_fun=metric_fun,
        metric=metric,
        metric_max=metric_max,
        learn_transform=transform.define_transform,
        apply_transform=apply_transformation,
        revert_transform=revert_transform,
        transform_parameter=transforms,
        workers=args.workers,
        debug=args.debug,
    )


def combine_transformations(
    data: DataLoaderT, transform: TransformDict, model_transform: Callable | None = None
) -> DataLoaderT:
    if data.properties["y_names"][0].endswith("toy"):
        transform["x"]["mean"][:] = 0
        transform["x"]["std"][:] = 1
        transform["y"]["mean"][:] = 0
        transform["y"]["std"][:] = 1
    data = set_transform(data, transform)
    data.add_transform(model_transform, optimizable=True)
    return data


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default="", type=str)
    parser.add_argument(
        "--dimension",
        choices=[
            "mosi/sentiment",
            "mosei/sentiment",
            "mosei/happiness",
            "iemocap/valence",
            "iemocap/arousal",
            "sewa/valence",
            "sewa/arousal",
            "recola/valence",
            "recola/arousal",
            "umeme/valence",
            "umeme/arousal",
            "tpot/constructs",
            "vreed/av",
            "toy/toy",
            "natoy/toy",
        ],
        default="mosi/sentiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "ehgr", "smurf"],
        default="base",
    )
    parser.add_argument(
        "--modalities",
        choices=[
            "a",
            "l",
            "v",
            "al",
            "av",
            "lv",
            "ah",
            "alv",
            "a+lv",
            "h",
            "hv",
            "s",
            "sv",
            "hs",
            "ahv",
            "ahsv",
            "hsv",
            "m",
        ],
        default="alv",
    )
    parser.add_argument("--fs", default="")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--interactions", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    folder = Path("experiments") / namespace_as_string(
        args, exclude=("workers", "debug")
    ).replace("/", "")
    train(get_partitions(args.dimension, 256), folder, args)
