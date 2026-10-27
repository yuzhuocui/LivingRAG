from dataclasses import dataclass, field
from src.utils import LLM_Model


@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False


    ner_type: str = "spacy"


    llm_ner_model: str = "qwen/qwen3.5-9b"


    llm_ner_seed_top_k: int = 3
    llm_ner_seed_max_degree: int = 300


    passage_length_norm_b: float = 0.0

    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])


@dataclass
class LivingRAGConfig(LinearRAGConfig):


    exp_alpha: float = 0.6
    exp_top_k: int = 5


    exp_beta: float = 1.0
    exp_tau_s: float = 0.3
    exp_mu: float = 0.01
    exp_epsilon: float = 0.05
    fusion_epsilon: float = 0.05


    use_temporal_decay: bool = False


    exp_activation_top_n: int = 0


    living_family_serial: bool = False


    tau_nli: float = 0.7
    tau_gs: float = 0.6
    tau_novel: float = 0.7
    max_retries: int = 2


    tau_merge: float = 0.9


    consolidate_interval: int = 0


    max_sentences: int = 10


    scaffold_template_weight: float = 0.0


    gs_top_k_evidence: int = 5


    nli_model_path: str = "cross-encoder/nli-deberta-v3-large"


    exp_db_path: str = ""
