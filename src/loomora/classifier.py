# Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.
# Original notices for adapted implementations are in licenses/.
"""Binary classifier using the transplanted classification inference chain."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d
from tqdm.auto import tqdm

from .inference_errors import handle_oom_errors
from .inference import load_stage_model
from .model import LoomoraModel
from .inference_config import InferenceConfig, get_default_inference_config
from .inference_constants import MemorySavingMode, XType, YType, SKLEARN_16_DECIMAL_PRECISION, PROBABILITY_EPSILON_ROUND_ZERO
from .inference_utils import infer_devices, determine_precision, infer_random_state, create_inference_engine, expand_dates_and_text, resolve_categorical_features_indices, resolved_n_estimators, resolved_softmax_temperature, balance_probas_by_class_counts
from .inference_validation import extract_input_shape, check_input_shape_matches, ensure_compatible_fit_inputs, ensure_compatible_predict_input_sklearn
from .downsample_correction import apply_class_weights, context_class_prior, downsample_class_weights
from .preprocessing.configs import ClassifierEnsembleConfig, FeatureSubsamplingMethod, SampleSubsamplingMethod
from .preprocessing.ensemble import LoomoraEnsemblePreprocessor, generate_classification_ensemble_configs, scale_n_estimators_for_feature_coverage
from .preprocessing.clean import clean_data, clean_data_transform
from .preprocessing.datamodel import FeatureModality
from .preprocessing.modality_detection import detect_feature_modalities
from .preprocessing.label_encoder import LoomoraLabelEncoder


class LoomoraClassifier(ClassifierMixin, BaseEstimator):
    """Binary and one-vs-rest multiclass prediction with Loomora V1.0."""

    estimator_type = "classifier"

    def __init__(
        self, model_path: str | Path | LoomoraModel, device="auto", *,
        n_estimators: int | Literal["auto"] = 8, random_state=42, categorical_features_indices=None,
        fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = "fit_with_cache",
        memory_saving_mode: MemorySavingMode = "auto", inference_precision="auto",
        n_preprocessing_jobs: int = 1, keep_cache_on_device: bool = False,
        kv_cache_precision=None, ignore_pretraining_limits: bool = False,
        softmax_temperature="auto", balance_probabilities: bool = False,
        average_before_softmax: bool = False, inference_config: dict | None = None,
        show_progress_bar: bool = False,
        inference_profile: Literal["memory_save", "speed"] = "memory_save", estimator_batch_size: int = 8,
    ):
        self.model_path, self.device = model_path, device
        self.n_estimators, self.random_state = n_estimators, random_state
        self.categorical_features_indices = categorical_features_indices
        self.fit_mode, self.memory_saving_mode = fit_mode, memory_saving_mode
        self.inference_precision, self.n_preprocessing_jobs = inference_precision, n_preprocessing_jobs
        self.keep_cache_on_device, self.kv_cache_precision = keep_cache_on_device, kv_cache_precision
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.softmax_temperature, self.balance_probabilities = softmax_temperature, balance_probabilities
        self.average_before_softmax, self.inference_config = average_before_softmax, inference_config
        self.show_progress_bar = show_progress_bar
        self.inference_profile, self.estimator_batch_size = inference_profile, estimator_batch_size

    def _initialize_model_variables(self) -> int:
        if self.inference_profile not in ("memory_save", "speed"):
            raise ValueError("inference_profile must be 'memory_save' or 'speed'")
        if not isinstance(self.estimator_batch_size, (int, np.integer)) or self.estimator_batch_size < 1:
            raise ValueError("estimator_batch_size must be a positive integer")
        if self.inference_profile == "speed" and self.fit_mode not in ("fit_preprocessors", "fit_with_cache"):
            raise ValueError("speed requires fit_preprocessors or fit_with_cache")
        if self.fit_mode not in ("low_memory", "fit_preprocessors", "fit_with_cache"):
            raise ValueError("fit_mode must be 'low_memory', 'fit_preprocessors' or 'fit_with_cache'")
        if self.n_estimators != "auto" and (not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1):
            raise ValueError("n_estimators must be a positive integer")
        self.devices_ = infer_devices(self.device)
        self.use_autocast_, self.forced_inference_dtype_, byte_size = determine_precision(self.inference_precision, self.devices_)
        self.inference_config_ = get_default_inference_config().override_with_user_input_and_resolve_auto(self.inference_config)
        if self.inference_config_.MAX_NUMBER_OF_CLASSES != 2:
            raise ValueError("Loomora supports binary classification; MAX_NUMBER_OF_CLASSES must be 2")
        self.model = self.model_path.eval() if isinstance(self.model_path, LoomoraModel) else load_stage_model(self.model_path)
        self.models_ = [self.model]
        self.model_version_ = self.model.model_version
        self.softmax_temperature_ = self.inference_config_.SOFTMAX_TEMPERATURE if self.softmax_temperature == "auto" else self.softmax_temperature
        self.auto_scale_n_estimators = True
        return byte_size

    def get_inference_config(self) -> InferenceConfig:
        return copy.deepcopy(self.inference_config_)


    def _initialize_dataset_preprocessing(
        self,
        X: XType,
        y: YType,
        random_state: int | np.random.Generator,
    ) -> tuple[list[ClassifierEnsembleConfig], np.ndarray, np.ndarray]:
        """Initialize the model for standard input."""
        # feature_names_in_/n_features_in_ have to describe what the caller passed,
        # not the wider frame expansion can make of it, so they come off the raw
        # input here, before any conversion.
        self.feature_names_in_, self.n_features_in_ = extract_input_shape(X)

        categorical_indices = resolve_categorical_features_indices(
            X, self.categorical_features_indices
        )
        X, date_transformer, text_transformer, feature_names, categorical_indices = expand_dates_and_text(X, categorical_features_indices=categorical_indices, inference_config=self.inference_config_)

        # Data validation and cleaning
        X, y, original_y_name = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=self,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            max_cpu_samples=self.inference_config_.MAX_CPU_SAMPLES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            ensure_y_numeric=False,
            devices=self.devices_,
        )

        feature_schema = detect_feature_modalities(
            X=X,
            feature_names=feature_names,
            provided_categorical_indices=categorical_indices,
            min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
            min_cardinality_for_text=self.inference_config_.MIN_CARDINALITY_FOR_TEXT,
        )
        X, ordinal_encoder, feature_schema = clean_data(
            X=X,
            feature_schema=feature_schema,
            passthrough_inf=self.get_inference_config().PASSTHROUGH_INF,
        )
        self.inferred_feature_schema_ = feature_schema
        self.ordinal_encoder_ = ordinal_encoder
        self.date_transformer_ = date_transformer
        self.text_transformer_ = text_transformer
        self.categorical_features_indices_ = categorical_indices
        self.n_train_samples_ = len(X)

        # Label encoding
        self.label_encoder_ = LoomoraLabelEncoder(original_target_name=original_y_name)
        y, label_metadata = self.label_encoder_.fit_transform(
            y=y, max_num_classes=self.inference_config_.MAX_NUMBER_OF_CLASSES
        )
        self.classes_ = label_metadata.classes
        self.n_classes_ = label_metadata.n_classes
        self.class_counts_ = label_metadata.class_counts

        # Ensemble definition
        preprocessor_configs = self.inference_config_.PREPROCESS_TRANSFORMS
        self.n_estimators_ = scale_n_estimators_for_feature_coverage(
            n_estimators=resolved_n_estimators(self),
            n_total_features=feature_schema.num_columns,
            preprocessor_configs=preprocessor_configs,
            auto_scale_n_estimators=self.auto_scale_n_estimators,
        )
        ensemble_configs = generate_classification_ensemble_configs(
            num_estimators=self.n_estimators_,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
            preprocessor_configs=preprocessor_configs,
            class_shift_method=self.inference_config_.CLASS_SHIFT_METHOD,
            n_classes=self.n_classes_,
            random_state=random_state,
            num_models=len(self.models_),
            outlier_removal_std=self.inference_config_.get_resolved_outlier_removal_std(
                estimator_type=self.estimator_type
            ),
            passthrough_inf=self.get_inference_config().PASSTHROUGH_INF,
        )
        assert len(ensemble_configs) == self.n_estimators_

        return ensemble_configs, X, y


    def fit(self, X: XType, y: YType) -> Self:
        """Prepare labeled context for binary or single-label multiclass prediction."""
        labels = LabelEncoder()
        encoded = labels.fit_transform(column_or_1d(y))
        if len(labels.classes_) <= 2:
            self.__dict__.pop("_ovr_classifiers_", None)
            return self._fit_binary(X, y)

        self._initialize_model_variables()
        params = self.get_params(deep=False)
        params["model_path"] = self.model
        classifiers = []
        for index in range(len(labels.classes_)):
            binary = LoomoraClassifier(**params)
            binary._fit_binary(X, (encoded == index).astype(np.int64))
            classifiers.append(binary)

        self._ovr_classifiers_ = classifiers
        self.classes_ = labels.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = classifiers[0].n_features_in_
        if hasattr(classifiers[0], "feature_names_in_"):
            self.feature_names_in_ = classifiers[0].feature_names_in_
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    @config_context(transform_output="default")  # type: ignore
    def _fit_binary(self, X: XType, y: YType) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.

        Returns:
            self
        """

        static_seed, _ = infer_random_state(self.random_state)
        byte_size = self._initialize_model_variables()
        ensemble_configs, X, y = self._initialize_dataset_preprocessing(
            X=X, y=y, random_state=static_seed
        )
        self.ensemble_configs_ = ensemble_configs

        # A numeric float32 table can be shared by fitted estimator pipelines.
        # Keep one owned snapshot so later writes to the caller's array stay private.
        if X.dtype == np.float32 and self.inference_config_.ENABLE_GPU_PREPROCESSING:
            X = X.copy(order="C")

        self.ensemble_preprocessor_ = LoomoraEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X.shape[0],
            feature_schema=self.inferred_feature_schema_,
            # Note: we use the static_seed so we're independent of the random generation
            # inside the initialize function above
            random_state=static_seed,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            keep_fitted_cache=(self.fit_mode == "fit_with_cache"),
            enable_gpu_preprocessing=self.inference_config_.ENABLE_GPU_PREPROCESSING,
            feature_subsampling_method=FeatureSubsamplingMethod(
                self.inference_config_.FEATURE_SUBSAMPLING_METHOD
            ),
            constant_feature_count=self.inference_config_.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT,
            subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES,
            sample_subsampling_method=SampleSubsamplingMethod(
                self.inference_config_.SAMPLE_SUBSAMPLING_METHOD
            ),
            importance_top_k_count=self.inference_config_.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
            X_train=X,
            y_train=y,
            task_type=self.estimator_type,
        )

        self._compute_downsample_correction(y_encoded=y)

        self.executor_ = create_inference_engine(
            fit_mode=self.fit_mode,
            X_train=X,
            y_train=y,
            models=self.models_,
            ensemble_preprocessor=self.ensemble_preprocessor_,
            devices_=self.devices_,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            task_type="multiclass",
            inference_mode=True,
            keep_cache_on_device=self.keep_cache_on_device,
            kv_cache_precision=self.kv_cache_precision,
            inference_profile=self.inference_profile,
            estimator_batch_size=self.estimator_batch_size,
        )

        print(f"model_version={self.model_version_} n_estimators={self.n_estimators_} fit_mode={self.fit_mode} inference_profile={self.inference_profile}", flush=True)
        return self


    def _compute_downsample_correction(self, *, y_encoded: np.ndarray) -> None:
        """Set `downsample_correction_weights_` from the fitted row sampler.

        Requires `ensemble_preprocessor_` and `n_classes_`. Uses `class_counts_`
        when the label encoder produced it, and the encoded labels otherwise (the
        differentiable-input path skips the encoder).
        """
        preprocessor = self.ensemble_preprocessor_
        if not preprocessor.downsample_shifted_prior:
            self.downsample_correction_weights_ = None
            return
        assert preprocessor.subsample_row_indices is not None
        train_counts = getattr(self, "class_counts_", None)
        if train_counts is None:
            train_counts = np.bincount(
                np.asarray(y_encoded).astype(np.int64), minlength=self.n_classes_
            )
        context_prior = context_class_prior(
            y_encoded, preprocessor.subsample_row_indices, self.n_classes_
        )
        self.downsample_correction_weights_ = downsample_class_weights(
            train_counts, context_prior
        )


    def _raw_predict(self, X, *, return_logits: bool, return_raw_logits: bool = False) -> torch.Tensor:
        check_is_fitted(self, "executor_")
        check_input_shape_matches(X, estimator=self)
        X = self.date_transformer_.transform(X)
        X = self.text_transformer_.transform(X)
        X = ensure_compatible_predict_input_sklearn(X, self)
        X = clean_data_transform(X, cat_indices=self.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL), ord_encoder=self.ordinal_encoder_, passthrough_inf=self.get_inference_config().PASSTHROUGH_INF)
        if any(stride < 0 for stride in X.strides):
            X = np.ascontiguousarray(X)
        with handle_oom_errors(self.devices_, X, model_type="classifier", n_train_samples=self.n_train_samples_):
            return self.forward(X, use_inference_mode=True, return_logits=return_logits, return_raw_logits=return_raw_logits)


    def _maybe_reweight_probas(self, probas: np.ndarray) -> np.ndarray:
        """Reweights the probabilities if a target_metric is specified.

        If a target metric is specified, the probabilities are reweighted based on
        the true holdout sets labels and predicted logits. This is done to tune the
        threshold for classification to the specified target metric.

        Args:
            probas: The predicted probabilities of the classes as a NumPy array.
                Shape (n_samples, n_classes).

        Returns:
            The input probas if no tuning is done, otherwise the reweighted
            probabilities.
        """
        if getattr(self, "tuned_classification_thresholds_", None) is None:
            return probas

        probas = probas / np.maximum(self.tuned_classification_thresholds_, 1e-8)
        return probas / probas.sum(axis=1, keepdims=True)


    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by the softmax temperature."""
        temp = resolved_softmax_temperature(self)
        if temp != 1.0:
            return logits / temp
        return logits


    def _average_across_estimators(self, tensors: torch.Tensor) -> torch.Tensor:
        """Averages a tensor across the estimator dimension (dim=0)."""
        return tensors.mean(dim=0)


    def _apply_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies the softmax function to the last dimension."""
        return torch.nn.functional.softmax(logits, dim=-1)


    def _apply_balancing(self, probas: torch.Tensor) -> torch.Tensor:
        """Applies class balancing to a probability tensor."""
        counts = getattr(self, "class_counts_", None)
        if counts is None:
            return probas
        return balance_probas_by_class_counts(probas, counts)


    def logits_to_probabilities(
        self,
        raw_logits: np.ndarray | torch.Tensor,
        *,
        softmax_temperature: float | None = None,
        average_before_softmax: bool | None = None,
        balance_probabilities: bool | None = None,
        downsample_correction_weights: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Convert logits to probabilities using the classifier's post-processing.

        Args:
            raw_logits: Logits with shape (n_estimators, n_samples, n_classes) or
                (n_samples, n_classes). If the logits have three dimensions, they are
                averaged across the estimator dimension (dim=0).
            softmax_temperature: Optional override for temperature scaling.
            average_before_softmax: Optional override for averaging order.
            balance_probabilities: Optional override for probability balancing.
            downsample_correction_weights: Optional override for the fitted
                `downsample_correction_weights_`, either one vector or one row
                per sample. `None` means "use the fitted weights"; tuning always
                passes explicit weights (ones where a fold applied no
                correction) because its folds carry their own correction.

        Returns:
            Probabilities with shape (n_samples, n_classes).
        """
        raw_logits = (
            raw_logits
            if isinstance(raw_logits, torch.Tensor)
            else torch.from_numpy(np.asarray(raw_logits))
        )
        used_temperature = (
            resolved_softmax_temperature(self)
            if softmax_temperature is None
            else softmax_temperature
        )
        use_average_before_softmax = (
            self.average_before_softmax
            if average_before_softmax is None
            else average_before_softmax
        )
        use_balance = (
            self.balance_probabilities
            if balance_probabilities is None
            else balance_probabilities
        )

        steps: list[Callable[[torch.Tensor], torch.Tensor]] = []

        if used_temperature != 1.0:

            def apply_temp(t: torch.Tensor) -> torch.Tensor:
                return t / used_temperature

            steps.append(apply_temp)

        if raw_logits.ndim >= 3:
            if use_average_before_softmax:
                steps.append(self._average_across_estimators)
                steps.append(self._apply_softmax)
            else:
                steps.append(self._apply_softmax)
                steps.append(self._average_across_estimators)
        elif raw_logits.ndim == 2:
            steps.append(self._apply_softmax)
        else:
            raise ValueError(
                f"Expected logits with 2 or more dims, got {raw_logits.ndim}"
            )

        # Restore the training prior first, so balancing (if requested) acts on
        # probabilities that express the training prior rather than the context's.
        correction_weights = (
            getattr(self, "downsample_correction_weights_", None)
            if downsample_correction_weights is None
            else downsample_correction_weights
        )
        if correction_weights is not None:

            def apply_correction(probas: torch.Tensor) -> torch.Tensor:
                return apply_class_weights(probas, correction_weights)

            steps.append(apply_correction)
        if use_balance:
            steps.append(self._apply_balancing)

        output = raw_logits
        for fn in steps:
            output = fn(output)

        return output


    def forward(self, X, *, use_inference_mode: bool = False, return_logits: bool = False, return_raw_logits: bool = False) -> torch.Tensor:
        if self.fit_mode == "fit_preprocessors":
            # Don't enable inference mode when differentiable_input=True (prompt tuning)
            # to allow gradients to flow through
            actual_inference_mode = use_inference_mode and not False
            self.executor_.use_torch_inference_mode(use_inference=actual_inference_mode)

        outputs = []
        for output, config in tqdm(
            self.executor_.iter_outputs(
                X,
                autocast=self.use_autocast_,
                task_type="multiclass",
            ),
            total=self.n_estimators_,
            desc="Loomora inference",
            unit="estimator",
            disable=not self.show_progress_bar,
        ):
            # Upcast from autocast's reduced precision so the post-processing
            # (temperature scaling, softmax, estimator averaging) runs in
            # float32, keeping predict_proba consistent with predict_logits.
            output = output.float()  # noqa: PLW2901
            original_ndim = output.ndim

            # This block correctly handles both single configs and lists of configs
            if original_ndim == 2:
                # Shape is [Nsamples, NClasses] -> [Nsamples, 1,  NClasses]
                processed_output = output.unsqueeze(1)
                config_list = [config]
            elif original_ndim == 3:
                # Shape is [Nsamples, batch_size, NClasses]
                processed_output = output
                config_list = config
            else:
                raise ValueError(
                    f"Output tensor must be 2d or 3d, got {original_ndim}d"
                )

            # Process the config_list (which is now guaranteed to be a list)
            output_batch = []
            for i, batch_config in enumerate(config_list):
                assert isinstance(batch_config, ClassifierEnsembleConfig)
                # If class_permutation is None - class shifting is disabled
                # So we slice to self.n_classes_ to ensure the output tensor matches
                # the expected number of classes
                if batch_config.class_permutation is None:
                    output_batch.append(processed_output[:, i, : self.n_classes_])
                else:
                    # make sure the processed_output num_classes are the same.
                    if len(batch_config.class_permutation) != self.n_classes_:
                        use_perm = np.arange(self.n_classes_)
                        use_perm[: len(batch_config.class_permutation)] = (
                            batch_config.class_permutation
                        )
                    else:
                        use_perm = batch_config.class_permutation

                    output_batch.append(processed_output[:, i, use_perm])

            outputs.append(torch.stack(output_batch, dim=1))

        # --- Post-processing ---
        stacked_outputs = torch.stack(outputs)

        if return_logits:
            temp_scaled = self._apply_temperature(stacked_outputs)
            output = self._average_across_estimators(temp_scaled)
        elif return_raw_logits:
            output = stacked_outputs
        else:
            output = self.logits_to_probabilities(stacked_outputs)

        # --- Final output shaping ---
        # Standard inference squeezes the singleton batch dimension.
        if output.ndim > 2 and use_inference_mode:
            output = output.squeeze(1) if not return_raw_logits else output.squeeze(2)

        if not use_inference_mode:
            # This case is primarily for fine-tuning where NLLLoss expects [B, C, N]
            if output.ndim == 2:  # was likely [N, C]
                output = output.unsqueeze(0)  # [1, N, C]
            output = output.transpose(0, 1).transpose(1, 2)

        return output


    def predict(self, X: XType) -> np.ndarray:
        """Predict the class labels for the provided input samples.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted class labels as a NumPy array.
        """
        probas = self._predict_proba(X=X)
        return self.classes_[np.argmax(probas, axis=1)]


    def predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        This is a wrapper around the `_predict_proba` method.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted probabilities of the classes as a NumPy array.
            Shape (n_samples, n_classes).
        """
        return self._predict_proba(X)


    @config_context(transform_output="default")  # type: ignore
    def _predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted probabilities of the classes as a NumPy array.
            Shape (n_samples, n_classes).
        """
        if hasattr(self, "_ovr_classifiers_"):
            scores = np.column_stack([classifier._predict_proba(X)[:, 1] for classifier in self._ovr_classifiers_])
            totals = scores.sum(axis=1, keepdims=True)
            probabilities = np.full_like(scores, 1 / len(self.classes_))
            return np.divide(scores, totals, out=probabilities, where=totals > 0)

        probas = (
            self._raw_predict(X, return_logits=False).float().detach().cpu().numpy()
        )
        probas = self._maybe_reweight_probas(probas=probas)
        if self.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
            probas = np.around(probas, decimals=SKLEARN_16_DECIMAL_PRECISION)
            probas = np.where(probas < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, probas)

        # Ensure probabilities sum to 1 in case of minor floating point inaccuracies
        # going from torch to numpy
        return probas / probas.sum(axis=1, keepdims=True)  # type: ignore
