class Config_eval:
    def __init__(self):
        super(Config_eval, self).__init__()
        self.loss_type      = 'npo' # change this with the experiment types provided above
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct'#'Qwen/Qwen2.5-3B-Instruct' 
        self.max_length     = 512
        self.selection       ='coreset' #high, ri, or emb or no (selection for impair)
        self.dataset       = 'bio'
        self.model_name     = 'llama'
        self.adaptor_path   = f'/home/praveen/eacl/outputs/unlearning/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}_model'
        self.retriever_model = 'thenlper/gte-small' 
        self.forget_path    = f'/home/praveen/eacl/data/{self.model_name}_{self.dataset}_forget.parquet'
        self.test_path    = f'/home/praveen/eacl/data/{self.model_name}_{self.dataset}_test.parquet'
        self.audit_path    = f'/home/praveen/eacl/data/{self.model_name}_{self.dataset}_audit.parquet'
        self.retain_path    = f'/home/praveen/eacl/data/retain_sets/{self.model_name}_{self.dataset}_{self.selection}.parquet'
