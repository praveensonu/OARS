class Config_gd:
    def __init__(self):
        super(Config_gd, self).__init__()
        self.loss_type      = 'gd' # change this with the experiment types provided above
        self.access_token   = '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' 
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj']
        self.batch_size     = 2
        self.gradient_accumulation_steps = 2 #always batch size of 8
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.selection       ='nnompnp'
        self.dataset       = 'muse'
        self.adaptor_path    = 'praveensonu/llama_muse' # this will be set based on the dataset in the code below
        self.save_dir       = f'/home/praveen/nnomp/outputs/ablation/key_modules/llama/{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = '/home/praveen/nnomp/data/muse/nnomp_forget.parquet'
        self.retain_path    = '/home/praveen/nnomp/data/muse/nnompnp_muse_retain.parquet'
        self.max_steps      = 200
        self.save_steps     = 20

class Config_snpo:
    def __init__(self):
        super(Config_snpo, self).__init__()
        self.loss_type      = 'snpo' # change this with the experiment types provided above
        self.access_token   = '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' 
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct'#'Qwen/Qwen2.5-3B-Instruct'#'meta-llama/Llama-3.1-8B-Instruct' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj']
        self.batch_size     = 2
        self.gradient_accumulation_steps = 2 #always batch size of 8
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.selection       ='random1'
        self.dataset       = 'bio'
        self.adaptor_path    = 'praveensonu/llama_bio'
        self.save_dir       = f'/home/praveen/nnomp/outputs/ablation/key_modules/llama/{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = '/home/praveen/nnomp/data/bio/nnomp_forget.parquet'
        self.retain_path    = '/home/praveen/nnomp/data/bio/llama_gracenp_bio_retain_random_not_omp_seed1.parquet'
        self.max_steps      = 200
        self.save_steps     = 20
        self.delta          = 0.0
        self.beta           = 3.5


class Config_npo:
    def __init__(self):
        super(Config_npo, self).__init__()
        self.loss_type      = 'npo' # change this with the experiment types provided above
        self.access_token   = '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' 
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj']
        self.batch_size     = 1
        self.gradient_accumulation_steps = 8 #always batch size of 8
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.selection       ='nnompnp'
        self.dataset       = 'bio'
        self.adaptor_path    = 'praveensonu/llama_bio'
        self.save_dir       = f'/home/praveen/nnomp/outputs/ablation/key_modules/llama/{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = '/home/praveen/nnomp/data/bio/nnomp_forget.parquet'
        self.retain_path    = '/home/praveen/nnomp/data/bio/llama_gracenp_bio_retain.parquet'
        self.max_steps      = 200
        self.save_steps     = 20
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0


class Config_rmu:
    def __init__(self):
        super(Config_rmu, self).__init__()
        self.loss_type      = 'rmu' # change this with the experiment types provided above
        self.access_token   = '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' 
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj']
        self.batch_size     = 2
        self.gradient_accumulation_steps = 4 #always batch size of 8
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.selection       ='1cluster'
        self.dataset       = 'muse'
        self.adaptor_path    = 'praveensonu/llama_muse'
        self.save_dir       = f'/home/praveen/nnomp/outputs/ablation/key_modules/llama/{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = '/home/praveen/nnomp/data/muse/nnomp_forget.parquet'
        self.retain_path    = '/home/praveen/nnomp/data/muse/llama_1cluster_muse_retain.parquet'
        self.max_steps      = 200
        self.save_steps     = 20