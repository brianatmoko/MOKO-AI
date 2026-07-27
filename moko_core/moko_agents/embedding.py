from moko_agents.llm_engine import engine

def get_embedding(text: str) -> list[float]:
    return engine.get_embedding(text)
