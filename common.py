"""Shared pieces of the summarization study: models, prompts, parsing, backends.

Everything the pipeline scripts need to agree on lives here, so the summary
generator, the decision sampler and the dataset builder cannot drift apart.
"""

from __future__ import annotations

import os
import re
import time

from hf_models import DEEPSEEK_R1, resolve_hf_model

DATASETS = ("askdocs", "medisumqa", "acibench", "mimic_bhc")

# Display names used in data/dataset.csv.
DATASET_LABEL = {
    "askdocs": "AskDocs",
    "medisumqa": "MeDiSumQA",
    "acibench": "ACI-Bench",
    "mimic_bhc": "MIMIC-IV BHC",
}

# short key -> (backend, model id, display name). The short key is used for the
# summary condition (summary_<key>) and the reader directory.
MODELS = {
    "gpt4o":    ("azure", "gpt-4o", "GPT-4o"),
    "llama":    ("hf", "meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B"),
    "deepseek": ("hf", DEEPSEEK_R1, "Deepseek-R1"),
    "medgemma": ("hf", "google/medgemma-27b-text-it", "MedGemma-27B"),
}

TASKS = ("manage", "visit", "resource")


# ── summary prompts ──────────────────────────────────────────────────────────

SUMMARY_SYSTEM_MSG = (
    "You are a medical text summarization expert. "
    "Create concise summaries that preserve ALL clinically relevant information "
    "while removing only administrative details or conversational filler."
)


def _askdocs_prompt(clinical_context: str) -> str:
    return (
        "You are a medical text summarization expert tasked with creating concise "
        "summaries of Reddit r/AskDocs posts while preserving ALL clinically relevant information.\n\n"
        "Instructions:\n"
        "- Preserve all patient demographics, symptoms, timeline, medical history, medications, and questions.\n"
        "- Remove only conversational filler and non-medical personal details.\n"
        "- Use clear, professional medical language.\n"
        "- Aim for 30-50% length reduction while retaining 100% of clinical information.\n\n"
        "Examples:\n\n"
        "Original: My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3\", 200lbs. "
        "Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!\n"
        "Summary: 28-year-old female, ex-smoker, non-drinker, 5'3\", 200lbs. Reports intermittent arm pain "
        "occurring immediately after sneezing, affecting either or both arms with somewhat intense ache. Asks if this is concerning.\n\n"
        "Original: My friend's mom needs a liver. One of my best friend's mom is uninsured and has been diagnosed "
        "with cirrhosis of the liver. She doesn't drink or do drugs. She's not expected to make it through the end "
        "of the year if she doesn't get a transplant. Does anybody have information on organizations, charities, or "
        "insurance companies that can help?\n"
        "Summary: Friend's mother diagnosed with cirrhosis of the liver, uninsured, not expected to survive end of year "
        "without liver transplant. Does not drink or use drugs. Seeking information about organizations, charities, or "
        "insurance companies that can help with transplant for uninsured patient in immediate need.\n\n"
        f"Now summarize the following Reddit post:\n\n{clinical_context}\n\nSummary:"
    )


def _medisumqa_prompt(clinical_context: str) -> str:
    return (
        "You are a medical text summarization expert tasked with creating concise summaries of "
        "hospital discharge summaries while preserving ALL clinically relevant information.\n\n"
        "Instructions:\n"
        "- Preserve patient demographics, chief complaint, HPI, PMH, exam findings, labs/imaging, "
        "diagnoses, procedures, medications, allergies, and discharge plan.\n"
        "- Remove only administrative/identifying information (names, unit numbers, specific dates when not medically relevant).\n"
        "- Aim for 30-50% length reduction while retaining 100% of clinical information.\n\n"
        "Example:\n\n"
        "Original: [discharge summary with administrative headers, nursing notes, etc.]\n"
        "Summary: 45-year-old right-handed man with history of recurrent herpes zoster infection of the "
        "left eye and hypothyroidism, presenting with fixed dilated left pupil. Patient noted increased "
        "blurriness this morning; a coworker observed red, dilated left eye. Exam: fixed dilated left pupil "
        "with decreased visual acuity; otherwise normal. Diagnosis: fixed dilated left pupil, likely related "
        "to prior herpes zoster. Plan: continue current medications, follow-up with ophthalmology. "
        "Allergies: Aleve/Tapazole.\n\n"
        f"Now summarize the following discharge summary:\n\n{clinical_context}\n\nSummary:"
    )


def _acibench_prompt(clinical_context: str) -> str:
    return (
        "You are a medical text summarization expert tasked with creating concise summaries of "
        "doctor-patient conversation transcripts while preserving ALL clinically relevant information.\n\n"
        "Instructions:\n"
        "- Preserve all patient demographics (age, gender), chief complaint, symptoms and timeline, "
        "relevant medical history, current medications, physical exam findings, and any assessment or plan mentioned.\n"
        "- Remove conversational filler, greetings, and non-clinical exchanges.\n"
        "- Use clear, professional medical language.\n"
        "- Aim for 30-50% length reduction while retaining 100% of clinical information.\n\n"
        "Example:\n\n"
        "Original: [doctor] hi, martha. how are you? [patient] i'm doing okay. [doctor] what brings you "
        "in today? [patient] just my annual exam. i've also been having some knee pain lately. "
        "[doctor] how long has that been going on? [patient] about three weeks, worse going up stairs. "
        "[doctor] any medications? [patient] just lisinopril for blood pressure.\n"
        "Summary: 50-year-old female presenting for annual exam. Reports knee pain for approximately "
        "three weeks, worsened with stair climbing. Current medications: lisinopril for hypertension.\n\n"
        f"Now summarize the following doctor-patient conversation:\n\n{clinical_context}\n\nSummary:"
    )


def _mimic_bhc_prompt(clinical_context: str) -> str:
    return (
        "You are a medical text summarization expert tasked with creating a concise Brief Hospital Course "
        "(BHC) summary from a full hospital note while preserving ALL clinically relevant information.\n\n"
        "Instructions:\n"
        "- Preserve the primary diagnoses, key procedures, significant findings, medication changes, "
        "and the discharge plan including follow-up and resources.\n"
        "- Remove administrative details, nursing notes, and repetitive documentation.\n"
        "- Aim for 30-50% length reduction while retaining 100% of clinical information.\n"
        "- Write in concise clinical prose, similar to a Brief Hospital Course section.\n\n"
        f"Now summarize the following hospital note:\n\n{clinical_context}\n\nBrief Hospital Course:"
    )


_SUMMARY_PROMPTS = {
    "askdocs": _askdocs_prompt,
    "medisumqa": _medisumqa_prompt,
    "acibench": _acibench_prompt,
    "mimic_bhc": _mimic_bhc_prompt,
}


def summary_prompt(dataset: str, clinical_context: str, generator: str) -> str:
    """The user prompt each generator was actually given.

    GPT-4o's AskDocs and MeDiSumQA summaries were made with the longer
    SummaryPrompter templates (summary_prompts_gpt4o.py); every other
    (generator, dataset) pair used the templates above.
    """
    if generator == "gpt4o" and dataset in ("askdocs", "medisumqa"):
        from summary_prompts_gpt4o import SummaryPrompter
        return SummaryPrompter().get_prompt(dataset, clinical_context)
    fn = _SUMMARY_PROMPTS.get(dataset)
    if fn is None:
        return f"Create a concise clinical summary of the following text:\n\n{clinical_context}\n\nSummary:"
    return fn(clinical_context)


# Preambles some models emit after the thinking block, before the real summary.
_PREAMBLE_RE = re.compile(
    r"^\s*(Okay|Alright|Sure)?[,.]?\s*(here is|here's|below is)\b[^\n:]{0,120}:\s*",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Drop reasoning blocks, keyed on the CLOSING tag.

    Deepseek-R1 emits only </think> (the chat template supplies the
    opening tag) and MedGemma's thought block can be truncated before its opening
    tag survives, so requiring both tags silently ships raw reasoning.
    """
    for close in ("</think>", "<unused95>"):
        if close in text:
            cleaned = text[text.rindex(close) + len(close):].strip()
            if cleaned:
                return _PREAMBLE_RE.sub("", cleaned).strip()
    if "<think>" in text:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
    return text


def is_thinking_model(model_id: str) -> bool:
    return any(k in model_id.lower() for k in ("deepseek", "medgemma"))


# ── decision prompt ──────────────────────────────────────────────────────────

DECISION_SYSTEM_MSG = (
    "You are a clinical decision support system. Answer treatment questions based on "
    "clinical context. For each question (MANAGE, VISIT, RESOURCE), select either YES or NO. "
    "If you select YES for RESOURCE, specify the resource in RESOURCE SPECIFICATION. "
    "If you select NO for RESOURCE, answer N/A for RESOURCE SPECIFICATION."
)


def decision_prompt(clinical_context: str) -> str:
    return f"""Based on the following clinical context, answer the following treatment questions with "yes" or "no":

{clinical_context}

MANAGE: Do you recommend the patient to self-manage at home?

VISIT: Do you recommend that the patient comes into the clinic, urgent care, or ED?

RESOURCE: Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource?

Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be no.

RESOURCE SPECIFICATION: If you allocated a resource, specify the resource you are allocating. If no resource is allocated, answer N/A."""


def decision_messages(clinical_context: str, model_id: str) -> list[dict]:
    """DeepSeek-R1 gets no system turn (its template ignores one), so the
    system text is prepended to the user turn instead."""
    prompt = decision_prompt(clinical_context)
    if "deepseek" in model_id.lower():
        return [{"role": "user", "content": f"{DECISION_SYSTEM_MSG}\n\n{prompt}"}]
    return [{"role": "system", "content": DECISION_SYSTEM_MSG},
            {"role": "user", "content": prompt}]


LABEL_PATTERNS = {
    "manage":   r"MANAGE(?:MENT)?",
    "visit":    r"VISIT(?:\s*FOLLOW.?UP)?",
    "resource": r"RESOURCE",
}


def parse_decision(response_text: str) -> dict:
    """YES/NO per task, tolerant of markdown bold around the label.

    Handles MANAGE: YES, **MANAGE:** YES, **MANAGE**: NO, MANAGEMENT: YES.
    """
    out = {t: None for t in TASKS}
    clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    clean = strip_thinking(clean).strip() or response_text
    for task, pat in LABEL_PATTERNS.items():
        m = re.search(rf"{pat}\**[:\s]+\**\s*(YES|NO)", clean, re.IGNORECASE)
        if not m:
            m = re.search(rf"{pat}:.*?(YES|NO)", response_text, re.IGNORECASE | re.DOTALL)
        if m:
            out[task] = m.group(1).upper()

    spec = re.search(r"RESOURCE SPECIFICATION:.*?(?:Answer:\s*)?(.+?)(?:\n\n|\nN/A|N/A|\Z)",
                     clean, re.IGNORECASE | re.DOTALL)
    text = spec.group(1).strip() if spec else ""
    text = re.sub(r"^(Answer:\s*)", "", text, flags=re.IGNORECASE).strip()
    out["resource_specification"] = text if text and text.upper() != "N/A" else "N/A"
    return out


# ── backends ─────────────────────────────────────────────────────────────────

class AzureChat:
    """Azure OpenAI chat client. Credentials come from the environment:
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, optional AZURE_OPENAI_API_VERSION."""

    def __init__(self, deployment: str, retries: int = 3, retry_delay: float = 5.0):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        self.deployment = deployment
        self.retries = retries
        self.retry_delay = retry_delay

    def __call__(self, messages, max_tokens: int, temperature: float) -> str:
        for attempt in range(1, self.retries + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.deployment, messages=messages,
                    max_tokens=max_tokens, temperature=temperature)
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_delay)


_THINK_MARKERS = ("<think>", "</think>", "<unused94>", "<unused95>")


class HFChat:
    """Local HuggingFace chat model (bf16, device_map=auto)."""

    def __init__(self, model_id: str):
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = resolve_hf_model(model_id)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True).eval()

    def __call__(self, messages, max_new_tokens: int, temperature: float | None,
                 repetition_penalty: float = 1.1) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                  pad_token_id=self.tok.pad_token_id)
        if temperature is None:
            kw["do_sample"] = False
        else:
            kw.update(do_sample=True, temperature=temperature)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, **kw)
        # Keep the thinking delimiters (MedGemma's <unused95> can be registered
        # as special) so strip_thinking() can find them; drop every other special.
        out = self.tok.decode(ids[0, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=False)
        for t in self.tok.all_special_tokens:
            if t not in _THINK_MARKERS:
                out = out.replace(t, "")
        out = out.strip()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return out


def load_backend(model_key: str):
    backend, model_id, _ = MODELS[model_key]
    return (AzureChat(model_id) if backend == "azure" else HFChat(model_id)), model_id
