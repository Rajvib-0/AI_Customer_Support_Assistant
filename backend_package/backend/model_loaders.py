"""Loads the embedding model and FLAN-T5 (imported lazily so the module can be imported cheaply)."""
import logging

from . import config as cfg

log = logging.getLogger("models")


def load_embedder():
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model %s", cfg.EMBEDDING_MODEL)
    return SentenceTransformer(cfg.EMBEDDING_MODEL)


def load_llm_generate():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    log.info("Loading LLM %s", cfg.LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.LLM_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log.info("LLM ready on %s", device)

    def generate(prompt: str, max_new_tokens: int = 128) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    return generate
