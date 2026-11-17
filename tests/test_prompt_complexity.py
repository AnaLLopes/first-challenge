import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_complexity import (
    LIMITES_NIVEIS,
    PESOS_COMPLEXIDADE,
    analisar_complexidade_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "eco-gemini.py"
SPEC = importlib.util.spec_from_file_location("eco_gemini_collector", COLLECTOR_PATH)
COLLECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COLLECTOR)


class PromptComplexityTests(unittest.TestCase):
    def test_requested_complexity_examples(self):
        examples = {
            "pergunta": "Qual é a capital do Brasil?",
            "tradução": "Traduza para inglês: Bom dia.",
            "cálculo": "Quanto é 125 × 48?",
            "função": "Crie uma função em Python que ordene uma lista de números.",
            "API": (
                "Desenvolva uma API REST em Python com autenticação, "
                "valide os dados de entrada e documente os endpoints."
            ),
            "múltiplas regras": (
                "Analise os dados primeiro, compare os resultados, depois "
                "valide cada condição e por fim apresente uma tabela com no "
                "máximo três recomendações, sem omitir exceções."
            ),
            "longo simples": (
                "Responda sim. "
                + "Este contexto apresenta informações gerais sobre o tema "
                "e conceitos comuns para leitura. " * 12
            ),
        }
        results = {
            name: analisar_complexidade_prompt(prompt)
            for name, prompt in examples.items()
        }

        self.assertEqual(results["pergunta"]["nivel"], "baixa")
        self.assertEqual(results["tradução"]["nivel"], "baixa")
        self.assertEqual(results["cálculo"]["nivel"], "baixa")
        self.assertEqual(results["função"]["nivel"], "moderada")
        self.assertEqual(results["API"]["nivel"], "alta")
        self.assertEqual(results["múltiplas regras"]["nivel"], "alta")
        self.assertEqual(results["longo simples"]["nivel"], "baixa")
        self.assertTrue(results["API"]["indicadores"]["conhecimento_tecnico"])
        self.assertTrue(results["API"]["indicadores"]["regras_e_restricoes"])
        self.assertTrue(
            results["múltiplas regras"]["indicadores"]["multiplas_etapas"]
        )

    def test_equal_length_prompts_can_have_different_complexity(self):
        simple = "Qual é 2+2?"
        reasoning = "Prove: 2+2?".ljust(len(simple))

        self.assertEqual(len(simple), len(reasoning))
        simple_result = analisar_complexidade_prompt(simple)
        reasoning_result = analisar_complexidade_prompt(reasoning)
        self.assertGreater(
            reasoning_result["pontuacao"],
            simple_result["pontuacao"],
        )
        self.assertTrue(reasoning_result["indicadores"]["raciocinio_logico"])

    def test_unrecognized_and_empty_prompts_are_still_analyzed(self):
        for prompt in ("", "palavras sem uma tarefa claramente identificável"):
            result = analisar_complexidade_prompt(prompt)
            self.assertEqual(result["status_analise"], "sucesso")
            self.assertIn(result["nivel"], (*LIMITES_NIVEIS, "muito_alta"))

    def test_weights_total_one_hundred(self):
        self.assertEqual(sum(PESOS_COMPLEXIDADE.values()), 100)

    def test_analysis_failure_is_recorded_without_blocking_request(self):
        class FakeModels:
            called = False

            def generate_content(self, **kwargs):
                self.called = True
                return SimpleNamespace(text="ok", usage_metadata=None)

        models = FakeModels()
        client = SimpleNamespace(models=models)
        item = {"prompt_id": "P001", "prompt": "Um prompt de teste."}

        with patch.object(
            COLLECTOR,
            "analisar_complexidade_prompt",
            side_effect=ValueError("falha simulada"),
        ):
            record = COLLECTOR.processar_prompt(item, 1, 1, client, "run")

        self.assertTrue(models.called)
        self.assertEqual(record["complexidade_prompt"]["status_analise"], "erro")
        self.assertEqual(
            record["complexidade_prompt"]["erro"]["tipo"],
            "ValueError",
        )
        self.assertEqual(record["status"], "sucesso_sem_dados_ambientais")


if __name__ == "__main__":
    unittest.main()
