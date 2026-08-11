class Config_ft:
    def __init__(self, model_name, dataset):

        # Inputs from command line
        self.model_name = model_name
        self.dataset = dataset

        # Fixed configurations
        self.LoRA_r = 16
        self.LoRA_alpha = 32
        self.LoRA_dropout = 0.05

        self.lr = 1e-4

        self.LoRa_targets = [
            "v_proj",
            "k_proj",
            "o_proj",
            "q_proj",
        ]

        self.batch_size = 2
        self.gradient_accumulation_steps = 16
        self.num_epochs = 10

        self.overwrite_dir = True
        self.weight_decay = 0.01

        self.max_length = 512

        # Model IDs
        model_ids = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "olmo": "allenai/OLMo-2-0425-1B-Instruct",
            "qwen": "Qwen/Qwen2.5-3B-Instruct",
        }

        self.model_id = model_ids[self.model_name]

        # Output directory
        self.save_dir = f"./outputs/{self.model_name}_{self.dataset}"

        # Dataset path
        if self.dataset == "bio":
            self.data_path = "./data/wmdp_bio.parquet"
        else:
            self.data_path = (
                f"./data/{self.model_name}_{self.dataset}_train.parquet"
            )



class Config_repair_auto:

    MODEL_IDS = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-3B-Instruct",
        "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    }

    def __init__(
        self,
        model_name,
        loss_type,
        dataset,
        selection,
    ):
        super(Config_repair_auto, self).__init__()

        # -------------------------------------------------
        # Experiment identifiers
        # -------------------------------------------------

        self.model_name = model_name
        self.model_id = self.MODEL_IDS[model_name]

        self.loss_type = loss_type
        self.dataset = dataset
        self.selection = selection

        # -------------------------------------------------
        # Training configuration
        # -------------------------------------------------

        self.lr = 1e-5

        self.batch_size = 2

        self.gradient_accumulation_steps = 4

        self.max_steps = 100

        self.save_steps = 20

        self.overwrite_dir = True

        self.max_length = 512

        # -------------------------------------------------
        # Unlearned adapter
        # -------------------------------------------------

        self.adaptor_path = (
            f"./outputs/unlearning/"
            f"{self.model_name}_{self.loss_type}_"
            f"{self.dataset}_no_model/"
            f"checkpoint-20"
        )

        # -------------------------------------------------
        # Repair output
        # -------------------------------------------------

        self.save_dir = (
            f"./outputs/repair/"
            f"{self.model_name}_{self.loss_type}_"
            f"{self.dataset}_{self.selection}"
        )

        # -------------------------------------------------
        # Repair dataset
        # -------------------------------------------------

        if self.selection in ["raslik", "emb", "random"]:

            self.data_path = (
                f"./data/retain_sets/"
                f"{self.model_name}_{self.dataset}_"
                f"{self.selection}.parquet"
            )

        elif self.selection == "oars":

            self.data_path = (
                f"./data/retain_sets/"
                f"{self.model_name}_{self.loss_type}_"
                f"{self.dataset}_{self.selection}.parquet"
            )

        else:
            raise ValueError(
                f"Unknown selection method: {self.selection}"
            )