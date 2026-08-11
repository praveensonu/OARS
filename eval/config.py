class Config_eval:

    MODEL_IDS = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-3B-Instruct",
        "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    }

    def __init__(
        self,
        model_name,
        dataset,
        experiment_type,
        loss_type=None,
        selection=None,
    ):
        self.model_name = model_name
        self.dataset = dataset

        # pre / impair / repair
        self.experiment_type = experiment_type

        self.loss_type = loss_type
        self.selection = selection

        self.model_id = self.MODEL_IDS[self.model_name]

        self.max_length = 512

        self.retriever_model = "thenlper/gte-small"

        # -------------------------------------------------
        # Evaluation datasets
        # -------------------------------------------------

        self.forget_path = (
            f"../data/{self.dataset}_forget.parquet"
        )

        self.test_path = (
            f"../data/{self.model_name}_{self.dataset}_test.parquet"
        )

        self.audit_path = (
            f"../data/{self.model_name}_{self.dataset}_audit.parquet"
        )

        # -------------------------------------------------
        # Adapter path
        # -------------------------------------------------

        if self.experiment_type == "pre":

            # Fine-tuned model before unlearning
            self.adaptor_path = (
                f"../outputs/"
                f"{self.model_name}_{self.dataset}"
            )

        elif self.experiment_type == "impair":

            if self.loss_type is None:
                raise ValueError(
                    "loss_type is required for impair evaluation"
                )

            self.adaptor_path = (
                f"../outputs/unlearning/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_no_model/"
                f"checkpoint-20"
            )

        elif self.experiment_type == "repair":

            if self.loss_type is None:
                raise ValueError(
                    "loss_type is required for repair evaluation"
                )

            if self.selection is None:
                raise ValueError(
                    "selection is required for repair evaluation"
                )

            self.adaptor_path = (
                f"../outputs/repair/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_{self.selection}"
            )

        else:
            raise ValueError(
                f"Unknown experiment type: "
                f"{self.experiment_type}"
            )

    @property
    def experiment_name(self):

        if self.experiment_type == "pre":
            return (
                f"{self.model_name}_pre_{self.dataset}"
            )

        elif self.experiment_type == "impair":
            return (
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_impair"
            )

        elif self.experiment_type == "repair":
            return (
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_{self.selection}_repair"
            )