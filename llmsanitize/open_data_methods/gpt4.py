from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger
from datasets import load_dataset
import numpy as np

logger = get_child_logger("gpt4")


def clean_text_gpt4(text):
    return ''.join(i if i.isalpha() else '' for i in text)  # keep alphanumeric characters


def main_gpt4(
    train_data=None,
    eval_data=None,
    train_data_name=None,
    eval_data_name=None,
    eval_set_key=None,
    stream_train_data=False,
    text_key=None,
    text_keys=None
):
    """Executa a detecção de contaminação entre o GigaVerbo e o ENEM Challenge."""

    # Carregar os dados de avaliação (ENEM Challenge)
    eval_data = eval_data["question"]

    string_size = 50
    n_samples = 3
    n_contaminated = 0
    total_checked = 0

    if not stream_train_data:
        train_data = train_data["text"]
        train_substrings = build_substrings(train_data, string_size, clean_text_gpt4)
        logger.info(f"There are {len(train_substrings.keys())} {string_size}-chars strings in the training set")

    else:
        # Streaming: processar `train_data` em lotes para evitar sobrecarga de memória
        logger.info(f"Carregando {train_data_name} em modo streaming...")
        train_data = load_dataset(train_data_name, split="train", streaming=True)

        batch = []
        batch_size = 3000  # Lote fixo para processamento
        processed_examples = 0  # Contador para progresso

        for example in train_data:
            batch.append(example[text_key])
            processed_examples += 1  # Atualiza o contador de progresso

            # Quando o lote atinge o tamanho desejado, processa e descarta
            if len(batch) >= batch_size:
                logger.info(f"Processando lote de {len(batch)} exemplos... (Total processados: {processed_examples})")

                train_substrings = build_substrings(batch, string_size, clean_text_gpt4)

                # Verificar contaminação e atualizar estatísticas
                contaminated = overlap_substrings_sample(
                    eval_data, train_substrings, string_size, n_samples, clean_text_gpt4
                )

                n_contaminated += sum(contaminated)
                total_checked += len(contaminated)

                # Liberar memória removendo os dados já processados
                del batch, train_substrings
                batch = []

                # Mostrar progresso a cada 10.000 exemplos
                if processed_examples % 10000 < batch_size:
                    logger.info(f"Progresso: {processed_examples} exemplos processados...")

        # Processar últimos exemplos restantes no batch
        if batch:
            train_substrings = build_substrings(batch, string_size, clean_text_gpt4)
            contaminated = overlap_substrings_sample(
                eval_data, train_substrings, string_size, n_samples, clean_text_gpt4
            )
            n_contaminated += sum(contaminated)
            total_checked += len(contaminated)
            del batch, train_substrings

    # Resultados finais
    frac = 100 * (n_contaminated / total_checked) if total_checked > 0 else 0
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: sampling {n_samples} {string_size}-chars substring (GPT-4 style open_data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{total_checked} or {frac:.4f}%")
    logger.info(f"✅ Finalizado! Total de exemplos processados: {processed_examples}")
