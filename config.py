class Config_ft:
    def __init__(self):
        super(Config_ft, self).__init__()
        self.model_id       = 'allenai/OLMo-2-0425-1B-Instruct'#'meta-llama/Llama-3.1-8B-Instruct'#'Qwen/Qwen2.5-3B-Instruct', allenai/OLMo-2-0425-1B-Instruct
        self.dataset        = 'muse'
        self.LoRA_r         = 16 #for muse we used 16, and bio 16
        self.LoRA_alpha     = 32 
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04 
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj'] # for llama 3.2 3b we used ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        self.batch_size     = 2 #32
        self.gradient_accumulation_steps = 16 #2 for finetuning
        self.num_epochs     = 10 #  10 epochs for finetuning
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.model_name    = 'olmo'
        self.save_dir       = f'./outputs/{self.model_name}_{self.dataset}' 
        self.max_length     = 512
        self.data_path      = f'./data/{self.model_name}_{self.dataset}_train.parquet'


class Config_repair:
    def __init__(self):
        super(Config_repair, self).__init__()
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct'
        self.loss_type      = 'snpo'
        self.selection       = 'ri'
        self.dataset        = 'muse'
        self.model_name    = 'qwen'
        self.adaptor_path    = f'./outputs/unlearning/{self.model_name}_{self.loss_type}_{self.dataset}_no_model'
        self.lr             = 1e-5
        self.batch_size     = 2
        self.gradient_accumulation_steps = 4
        self.max_steps      = 100
        self.save_steps     = 20
        self.overwrite_dir  = True
        self.save_dir       = f'./outputs/repair/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}'
        self.max_length     = 512
        self.data_path      = f'./data/retain/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}.parquet'


class Config_repair_auto: #for repair auto if we want to do multiple, use selection not model. it will autimatically run on the models
    MODEL_IDS = {
        'llama': 'meta-llama/Llama-3.1-8B-Instruct',
        'qwen': 'Qwen/Qwen2.5-3B-Instruct',
        'olmo': 'allenai/OLMo-2-0425-1B-Instruct',
    }

    def __init__(self, model_name='llama', loss_type='ga'):
        super(Config_repair_auto, self).__init__()
        self.model_name      = model_name
        self.model_id        = self.MODEL_IDS[model_name]
        self.loss_type       = loss_type
        self.selection        = 'randri'  # 'high'
        self.dataset         = 'bio'
        self.adaptor_path     = f'./outputs/unlearning/{self.model_name}_{self.loss_type}_{self.dataset}_no_model/checkpoint-20'
        self.lr              = 1e-5
        self.batch_size      = 2
        self.gradient_accumulation_steps = 4
        self.max_steps       = 100
        self.save_steps      = 20
        self.overwrite_dir   = True
        self.save_dir        = f'./outputs/repair/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}'
        self.max_length      = 512
        #self.data_path       = f'./data/retain_sets/{self.model_name}_{self.dataset}_{self.selection}.parquet'
        self.data_path      = f'./data/retain_sets/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}.parquet' #for representation shift
