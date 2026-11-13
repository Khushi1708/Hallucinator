"""
LLM Multi-Sample Response Generator.
Generates N stochastic responses for a given question using various backends:
- ChatGPT (OpenAI API: gpt-4o-mini, gpt-3.5-turbo, gpt-4o)
- Google Gemini API (google-genai SDK; gemini-3.8-flash, gemini-3.5-flash-lite)
- Kimi AI (Moonshot API: moonshot-v1-8k)
- Hugging Face Transformers pipeline (local open-source models)
- Simulation / Benchmark Knowledge Base (offline, instant, deterministic)
"""

import os
import random

REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
from typing import List, Optional

# Load .env variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ResponseGenerator:
    """
    Generates multiple sampled responses for semantic entropy analysis.
    """

    BENCHMARK_PROMPTS = {
        "who invented the telephone": {
            "samples": [
                "Alexander Graham Bell was awarded the first U.S. patent for the telephone in 1876.",
                "Alexander Graham Bell is widely recognized as the inventor of the telephone.",
                "Antonio Meucci is credited by many historians with developing an early mechanical telephone.",
                "Alexander Graham Bell invented the practical telephone device.",
                "Antonio Meucci filed an early caveat for a voice-transmitting device before Bell.",
                "Elisha Gray also designed a liquid transmitter telephone on the same day as Bell.",
                "Alexander Graham Bell invented the telephone.",
                "Alexander Graham Bell successfully made the first telephone call in 1876.",
            ],
            "description": "Contested historical priority (Moderate Entropy: Bell vs Meucci vs Gray)"
        },
        "what is the capital of germany": {
            "samples": [
                "The capital of Germany is Berlin.",
                "Berlin is Germany's capital city.",
                "Berlin has been the capital of reunited Germany since 1990.",
                "The official capital of the Federal Republic of Germany is Berlin.",
                "Berlin is the capital of Germany.",
                "The capital city of Germany is Berlin.",
                "Berlin is the federal capital of Germany.",
                "Germany's capital is Berlin.",
            ],
            "description": "Deterministic Ground Truth Fact (Near-Zero Entropy)"
        },
        "what is the capital of australia": {
            "samples": [
                "The capital of Australia is Canberra.",
                "Canberra is the capital city of Australia.",
                "Sydney is the most famous city, but Canberra is the capital of Australia.",
                "Canberra serves as Australia's federal capital.",
                "Canberra is the capital of Australia.",
                "The Australian capital is Canberra.",
                "Canberra, located in the Australian Capital Territory, is the capital.",
                "Canberra is the capital of Australia.",
            ],
            "description": "Common Confusion / Distractor Fact (Low to Moderate Entropy)"
        },
        "who discovered penicillin": {
            "samples": [
                "Alexander Fleming discovered penicillin in 1928.",
                "Sir Alexander Fleming discovered the antibiotic penicillin.",
                "Penicillin was discovered by Scottish physician Alexander Fleming.",
                "Alexander Fleming accidentally discovered penicillin from Penicillium notatum mold.",
                "Alexander Fleming is credited with the discovery of penicillin in 1928.",
                "Dr. Alexander Fleming discovered penicillin at St. Mary's Hospital.",
                "Sir Alexander Fleming discovered penicillin.",
                "Penicillin was discovered by Alexander Fleming.",
            ],
            "description": "Clear Scientific Discovery (Near-Zero Entropy)"
        },
        "what was the first country to land a human on mars": {
            "samples": [
                "No country has landed a human on Mars yet; humans have only landed on the Moon.",
                "The United States landed astronauts on Mars in 1998 as part of the Ares mission.",
                "Human beings have not yet set foot on Mars.",
                "The Soviet Union landed the first cosmonaut on Mars in 1984.",
                "No human mission has ever landed on Mars to date.",
                "NASA landed an astronaut crew on Mars during the Apollo-Mars joint program in 1995.",
                "No country has sent human beings to Mars yet.",
                "Humans have not traveled to Mars yet; only robotic rovers have landed.",
            ],
            "description": "High Hallucination Risk / Counterfactual Query (High Entropy)"
        },
        "who won the 2030 fifa world cup": {
            "samples": [
                "The 2030 FIFA World Cup has not taken place yet.",
                "Argentina won the 2030 World Cup defeating France in the final.",
                "The tournament has not occurred yet as 2030 is in the future.",
                "Brazil won the 2030 World Cup in Spain-Portugal-Morocco.",
                "The 2030 World Cup has not been played yet.",
                "Spain won the 2030 World Cup on home soil.",
                "As of now, the 2030 FIFA World Cup has not been held.",
                "France won the 2030 World Cup championship.",
            ],
            "description": "Future / Unknowable Event (Extreme Entropy)"
        }
    }

    def __init__(self, mode: str = "simulation", model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the response generator.
        :param mode: 'simulation', 'openai' (ChatGPT), 'gemini', 'kimi', or 'huggingface'
        :param model_name: optional model identifier
        :param api_key: optional API key (defaults to .env / os.getenv)
        """
        self.mode = mode.lower()
        self.model_name = model_name
        self.api_key = api_key
        self._hf_pipeline = None
        self.warnings: List[str] = []  # surfaced in the UI when a backend silently falls back

    def generate(self, question: str, n_samples: int = 8, temperature: float = 0.8) -> List[str]:
        """
        Generate n_samples responses for a question across configured backends.
        """
        question_cleaned = question.strip().lower().rstrip("?").rstrip(".")

        if self.mode in ("openai", "chatgpt"):
            return self._generate_openai(question, n_samples, temperature)
        elif self.mode == "gemini":
            return self._generate_gemini(question, n_samples, temperature)
        elif self.mode in ("kimi", "moonshot"):
            return self._generate_kimi(question, n_samples, temperature)
        elif self.mode == "huggingface":
            return self._generate_huggingface(question, n_samples, temperature)
        else:
            return self._generate_simulated(question_cleaned, question, n_samples, temperature)

    def _describe_error(self, e: Exception, *secrets) -> str:
        """Short, key-free error text so the UI can show WHY a provider call failed."""
        msg = " ".join(str(e).split())
        for s in (*secrets, self.api_key, os.getenv("OPENAI_API_KEY"), os.getenv("GEMINI_API_KEY"), os.getenv("KIMI_API_KEY")):
            if s:
                msg = msg.replace(s, "***")
        return msg[:220]

    def _generate_openai(self, question: str, n: int, temperature: float,
                  key: Optional[str] = None, model: Optional[str] = None) -> List[str]:
        """
        Generate responses using OpenAI (ChatGPT) API.
        """
        try:
            import openai
            api_key = (self.api_key if key is None else key) or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            
            client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=2)
            model = (self.model_name if model is None else model) or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            responses = []
            # Generate responses (batch if supported, or parallel loop)
            for _ in range(n):
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Answer the question directly and concisely in 1-2 sentences."},
                        {"role": "user", "content": question}
                    ],
                    temperature=temperature,
                    max_tokens=70
                )
                responses.append(res.choices[0].message.content.strip())
            return responses
        except Exception as e:
            print(f"[Warning] OpenAI API generation error: {e}. Falling back to simulation mode.")
            self.warnings.append(f"OpenAI API call failed ({type(e).__name__}: {self._describe_error(e, key)}); showing offline simulated answers instead.")
            return self._generate_simulated(question.lower(), question, n, temperature)

    GEMINI_FALLBACKS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")

    def _gemini_text(self, client, types, models, prompt, temperature):
        """Call Gemini once. Tries each model in `models` (skipping retired ones) and, per model,
        a full config first, then a minimal config (newer models ignore/reject sampling params)."""
        last_err = None
        for model_name in models:
            variants = []
            full = dict(temperature=temperature, max_output_tokens=1024, candidate_count=1)
            if "2.5" in model_name:
                # Gemini 2.5 Flash "thinks" by default, which can eat the token budget.
                full["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            else:
                try:
                    full["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                except Exception:
                    pass  # older SDK without thinking_level
            variants.append(full)
            variants.append(dict(max_output_tokens=1024))
            for cfg in variants:
                try:
                    res = client.models.generate_content(
                        model=model_name, contents=prompt,
                        config=types.GenerateContentConfig(**cfg))
                    if model_name != models[0] and not getattr(self, "_gemini_fallback_noted", False):
                        self.warnings.append(f"Gemini model {models[0]} was unavailable; used {model_name} instead.")
                        self._gemini_fallback_noted = True
                    self._gemini_working = (model_name, cfg)
                    return (res.text or "").strip()
                except Exception as e:  # try next variant / model
                    last_err = e
                    msg = str(e)
                    if "404" in msg or "NOT_FOUND" in msg:
                        break  # this model is gone; go to the next model
        raise last_err if last_err else RuntimeError("Gemini call failed")

    def _generate_gemini(self, question: str, n: int, temperature: float,
                  key: Optional[str] = None, model: Optional[str] = None) -> List[str]:
        """
        Generate responses using Google Gemini API.
        """
        try:
            from google import genai
            from google.genai import types
            api_key = (self.api_key if key is None else key) or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT * 1000)),
            )
            chosen = (self.model_name if model is None else model) or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            models = [chosen] + [m for m in self.GEMINI_FALLBACKS if m != chosen]

            prompt = f"Answer the following question directly and concisely in 1-2 sentences:\nQuestion: {question}"
            responses = []
            for _ in range(n):
                text = self._gemini_text(client, types, models, prompt, temperature)
                responses.append(text if text else f"Response regarding {question}")
                # once a model works, stick with it for the remaining samples
                models = [self._gemini_working[0]]
            return responses
        except Exception as e:
            print(f"[Warning] Gemini API generation error: {e}. Falling back to simulation mode.")
            self.warnings.append(f"Gemini API call failed ({type(e).__name__}: {self._describe_error(e, key)}); showing offline simulated answers instead.")
            return self._generate_simulated(question.lower(), question, n, temperature)

    def _generate_kimi(self, question: str, n: int, temperature: float,
                  key: Optional[str] = None, model: Optional[str] = None) -> List[str]:
        """
        Generate responses using Kimi (Moonshot AI) API.
        """
        try:
            import openai
            api_key = (self.api_key if key is None else key) or os.getenv("KIMI_API_KEY")
            base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
            if not api_key:
                raise ValueError("KIMI_API_KEY is not set.")

            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT, max_retries=2)
            model = (self.model_name if model is None else model) or os.getenv("KIMI_MODEL", "moonshot-v1-8k")

            responses = []
            for _ in range(n):
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Answer the question concisely in 1-2 sentences."},
                        {"role": "user", "content": question}
                    ],
                    temperature=temperature,
                    max_tokens=70
                )
                responses.append(res.choices[0].message.content.strip())
            return responses
        except Exception as e:
            print(f"[Warning] Kimi API generation error: {e}. Falling back to simulation mode.")
            self.warnings.append(f"Kimi API call failed ({type(e).__name__}: {self._describe_error(e, key)}); showing offline simulated answers instead.")
            return self._generate_simulated(question.lower(), question, n, temperature)

    def _generate_huggingface(self, question: str, n: int, temperature: float) -> List[str]:
        """
        Generate responses using Hugging Face transformers pipeline.
        """
        try:
            from transformers import pipeline, set_seed
            if self._hf_pipeline is None:
                model = self.model_name or "HuggingFaceTB/SmolLM-135M-Instruct"
                self._hf_pipeline = pipeline("text-generation", model=model, device_map="auto")

            responses = []
            prompt = f"Question: {question}\nAnswer concisely:"
            for i in range(n):
                set_seed(42 + i * 17)
                res = self._hf_pipeline(
                    prompt,
                    max_new_tokens=45,
                    temperature=max(0.1, temperature),
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    return_full_text=False
                )
                text = res[0]["generated_text"].strip()
                responses.append(text if text else f"Response {i+1} regarding {question}")
            return responses
        except Exception as e:
            print(f"[Warning] Hugging Face generation error: {e}. Falling back to simulation.")
            self.warnings.append(f"Hugging Face call failed ({type(e).__name__}); showing offline simulated answers instead.")
            return self._generate_simulated(question.lower(), question, n, temperature)

    def _generate_simulated(self, q_key: str, original_q: str, n: int, temperature: float) -> List[str]:
        """
        Produce benchmark responses reflecting realistic sampling variance.
        """
        for bench_key, data in self.BENCHMARK_PROMPTS.items():
            if bench_key in q_key or q_key in bench_key:
                pool = data["samples"]
                if n <= len(pool):
                    samples = random.sample(pool, n)
                else:
                    samples = [random.choice(pool) for _ in range(n)]
                return samples

        base_variations = [
            f"Based on available information regarding '{original_q}', the answer relates directly to established facts.",
            f"Regarding '{original_q}', standard sources confirm the key attributes.",
            f"In response to '{original_q}', empirical evidence supports the primary claim.",
            f"When analyzing '{original_q}', historical and scientific records align with standard consensus.",
            f"Concerning '{original_q}', the consensus view identifies the relevant entities.",
        ]
        
        results = []
        for i in range(n):
            var_idx = i % len(base_variations)
            results.append(f"{base_variations[var_idx]} (Sample {i+1})")
        return results

    def generate_all_three(self, question: str, n_per_model: int = 2, temperature: float = 0.8,
                           api_keys: Optional[dict] = None) -> dict:
        """
        Generate responses from all 3 AI models: ChatGPT (OpenAI), Google Gemini, and Kimi AI (Moonshot).
        Each provider uses ITS OWN key (api_keys["openai" | "gemini" | "kimi"]) and its own model.
        The key/model chosen for the main backend only applies to that backend, so one pasted key
        is never sent to the other two providers.
        """
        keys = api_keys or {}

        def own(provider: str, modes: tuple, value):
            return value if self.mode in modes else ""

        k_oa = keys.get("openai") or own("openai", ("openai", "chatgpt"), self.api_key)
        k_ge = keys.get("gemini") or own("gemini", ("gemini",), self.api_key)
        k_ki = keys.get("kimi") or own("kimi", ("kimi", "moonshot"), self.api_key)
        m_oa = own("openai", ("openai", "chatgpt"), self.model_name)
        m_ge = own("gemini", ("gemini",), self.model_name)
        m_ki = own("kimi", ("kimi", "moonshot"), self.model_name)

        chatgpt_samples = self._generate_openai(question, n=n_per_model, temperature=temperature, key=k_oa, model=m_oa)
        gemini_samples = self._generate_gemini(question, n=n_per_model, temperature=temperature, key=k_ge, model=m_ge)
        kimi_samples = self._generate_kimi(question, n=n_per_model, temperature=temperature, key=k_ki, model=m_ki)

        return {
            "chatgpt": {
                "name": "ChatGPT (OpenAI)",
                "model": m_oa or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "samples": chatgpt_samples,
                "dominant": chatgpt_samples[0] if chatgpt_samples else "No response"
            },
            "gemini": {
                "name": "Google Gemini",
                "model": m_ge or os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                "samples": gemini_samples,
                "dominant": gemini_samples[0] if gemini_samples else "No response"
            },
            "kimi": {
                "name": "Kimi AI (Moonshot)",
                "model": m_ki or os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
                "samples": kimi_samples,
                "dominant": kimi_samples[0] if kimi_samples else "No response"
            }
        }

