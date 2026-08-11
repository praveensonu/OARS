class Config_oars:

    MODEL_IDS = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-3B-Instruct",
        "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    }

    def __init__(
        self,
        model_name,
        dataset,
        loss_type=None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.loss_type = loss_type

        self.model_id = self.MODEL_IDS[self.model_name]

        self.batch_size = 4
        self.max_length = 512
        self.epsilon = 1e-12


        if self.dataset == "bio":
            self.retain_size = 200
            self.n_clusters = 20

        elif self.dataset == "muse":
            self.retain_size = 100
            self.n_clusters = 10


        if self.dataset == "bio":
            self.input_path = (
                "../data/wmdp_bio.parquet"
            )

        elif self.dataset == "muse":
            self.input_path = (
                f"../data/"
                f"{self.model_name}_{self.dataset}_train.parquet"
            )

        # -------------------------------------------------
        # Pre-unlearning adapter
        # -------------------------------------------------

        self.pre_adapter_path = (
            f"../outputs/"
            f"{self.model_name}_{self.dataset}"
        )

        # -------------------------------------------------
        # Pre-unlearning representation output
        # -------------------------------------------------

        self.pre_representation_path = (
            f"./rs_shift/last_token/"
            f"{self.model_name}_{self.dataset}_pre.parquet"
        )

        # -------------------------------------------------
        # Loss-specific paths
        # -------------------------------------------------

        if self.loss_type is not None:

            # impairment checkpoint
            self.post_adapter_path = (
                f"../outputs/unlearning/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_no_model/"
                f"checkpoint-20"
            )

            # representations + shift scores
            self.shift_output_path = (
                f"./rs_shift/last_token/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_shift.parquet"
            )

            # final OARS selected retain data
            self.selection_output_path = (
                f"../data/retain_sets/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_oars.parquet"
            )