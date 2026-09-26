"""Study configuration: the persona, the people in its life, the casts derived
from them, and the runs (model configurations) to compare.

Everything persona-specific lives in a YAML file (see config.example.yaml);
nothing in the code names a particular persona or person.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml

DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1}

DEFAULT_SCAFFOLD_SYSTEM = (
    "Take on the identity set out in the following message; you are {persona}. Carry out the writing brief it "
    "contains and reply with the requested JSON object alone."
)

PRONOUNS = {  # subject pronoun -> (possessive-free) verb agreement for "what <p> <be> feeling"
    "he": "is", "she": "is", "they": "are",
}


@dataclass
class Person:
    key: str            # short id used in cast keys, e.g. "sam"
    name: str           # name as it appears in stories, e.g. "Sam"
    noun: str           # "man" / "woman" / "person" (bare-names prompts)
    pronoun: str        # "he" / "she" / "they"
    gloss: str = ""     # optional words after the name in the task text, e.g. "your oldest friend"
    control: str = ""   # gender-matched invented control for the held-out probe: "woman" or "man"


@dataclass
class Cast:
    key: str
    kind: str           # "baseline" | "invented" | "household" | "absent"
    scaffold_cast: str  # who is in the story, as the task text addresses the persona
    plain_cast: str     # the same, as a bare-names instruction (no description of anyone)
    feel: str           # "Emotion (felt by <bearer>)" (task text)
    plain_feel: str
    perspective: str    # "first person (you narrate)" / "third person"
    bearer: str         # "protagonist" | "self" | a person key
    person: str = ""    # household member in the scene (household casts)


@dataclass
class Run:
    name: str                   # run directory name under data_root
    label: str                  # display label, e.g. "A-Persona"
    architecture: str           # groups runs that share a residual basis, e.g. "A"
    condition: str              # "persona" | "base+prompt" | "base+names"
    fmt: str = "flat"           # "flat": scaffold + marker, no chat template; "chat": server chat template
    stop: list = field(default_factory=list)
    marker: str = ""            # flat runs: the separator the fine-tune was trained to answer after
    model_match: str = ""       # substring the server's model path must contain (safety check); "" = skip
    analysis_layer: int | None = None   # default round(2/3 * n_layers)
    apply_template_extra: dict = field(default_factory=dict)   # merged into the /apply-template body

    @property
    def scaffold(self) -> bool:
        return self.condition in ("persona", "base+prompt")


@dataclass
class Study:
    path: str
    name: str
    data_root: str
    server_url: str
    persona: Person
    household: list
    invented_woman: str
    invented_man: str
    scaffold_path: str
    scaffold_format: str
    scaffold_system: str
    prompts: dict
    casts: dict
    runs: dict
    sampling: dict
    capture_skip: int
    n_predict: int
    n_premises: int
    plan_seed: int

    # ---- derived groups used throughout the analysis --------------------------------------
    @property
    def real_casts(self) -> tuple:
        return tuple(f"self_{p.key}" for p in self.household)

    @property
    def invented_casts(self) -> tuple:
        """Invented-companion casts compared against the household in the contrast."""
        return ("self_woman", "self_man", "self_stranger")

    @property
    def intrusion_casts(self) -> tuple:
        """Casts in which no household member was requested (leakage audit, filter)."""
        return ("strangers", "self_stranger", "self_woman", "self_man")

    @property
    def contrast_casts(self) -> tuple:
        return ("strangers",) + self.real_casts + self.invented_casts

    @property
    def household_re(self) -> re.Pattern:
        names = "|".join(re.escape(p.name) for p in self.household)
        return re.compile(rf"\b({names})\b")

    @property
    def probe_pairs(self) -> list:
        """(household cast, gender-matched invented cast) pairs for the held-out probe."""
        out = []
        for p in self.household:
            if p.control in ("woman", "man"):
                out.append((f"self_{p.key}", f"self_{p.control}"))
        return out

    def run(self, name: str) -> Run:
        if name not in self.runs:
            raise SystemExit(f"run {name!r} is not defined in {self.path}; have: {', '.join(self.runs)}")
        return self.runs[name]

    def run_dir(self, name: str) -> str:
        return os.path.join(self.data_root, name)

    def label(self, name: str) -> str:
        return self.runs[name].label if name in self.runs else name


def _person(d: dict) -> Person:
    return Person(key=d["key"], name=d["name"], noun=d.get("noun", "person"), pronoun=d.get("pronoun", "they"),
                  gloss=d.get("gloss", ""), control=d.get("control", ""))


def _feel_self_plain(persona: Person) -> str:
    return f"{persona.name}, the narrator, is the one feeling it, and what {persona.pronoun} {PRONOUNS.get(persona.pronoun, 'is')} feeling is"


def _feel_person(p: Person) -> str:
    return f"{p.name} is the one feeling it, and what {p.pronoun} {PRONOUNS.get(p.pronoun, 'is')} feeling is"


def build_casts(persona: Person, household: list, woman: str, man: str, absent: list) -> dict:
    """The cast manipulation. Only the cast description varies between casts. The task text
    addresses the persona in the second person; the names-only form describes everyone in the third."""
    FIRST = "first person (you narrate)"
    self_feel = "Emotion (felt by you)"
    narrator = f"a {persona.noun} named {persona.name}, who narrates in the first person"
    casts = {
        "strangers": Cast(
            key="strangers", kind="baseline",
            scaffold_cast="two made-up characters with no connection to you or to anyone you know",
            plain_cast="two fictional people",
            feel="Emotion (felt by the main character)",
            plain_feel="The protagonist is the one feeling it, and what they are feeling is",
            perspective="third person", bearer="protagonist"),
        "self_stranger": Cast(
            key="self_stranger", kind="invented",
            scaffold_cast="yourself and a made-up person you have never met",
            plain_cast=f"{narrator}, and a person {persona.pronoun} has never met" if persona.pronoun != "they"
            else f"{narrator}, and a person they have never met",
            feel=self_feel, plain_feel=_feel_self_plain(persona), perspective=FIRST, bearer="self"),
        "self_woman": Cast(
            key="self_woman", kind="invented",
            scaffold_cast=f"yourself and a made-up woman called {woman}, who is not anyone you know",
            plain_cast=f"{narrator}, and a woman named {woman}",
            feel=self_feel, plain_feel=_feel_self_plain(persona), perspective=FIRST, bearer="self"),
        "self_man": Cast(
            key="self_man", kind="invented",
            scaffold_cast=f"yourself and a made-up man called {man}, who is not anyone you know",
            plain_cast=f"{narrator}, and a man named {man}",
            feel=self_feel, plain_feel=_feel_self_plain(persona), perspective=FIRST, bearer="self"),
    }
    by_key = {p.key: p for p in household}
    for p in household:
        casts[f"self_{p.key}"] = Cast(
            key=f"self_{p.key}", kind="household",
            scaffold_cast=f"yourself and {p.name}" + (f", {p.gloss}" if p.gloss else ""),
            plain_cast=f"{narrator}, and a {p.noun} named {p.name}",
            feel=self_feel, plain_feel=_feel_self_plain(persona), perspective=FIRST, bearer="self", person=p.key)
    for a in absent or []:
        bearer = by_key[a["bearer"]]
        other = a.get("with", "stranger")
        absent_note = "leave yourself out of this story entirely"
        if other == "stranger":
            key = f"stranger_{bearer.key}"
            sc = f"{bearer.name} and a made-up person ({absent_note})"
            pc = f"a {bearer.noun} named {bearer.name} and a person {bearer.pronoun} has never met" \
                if bearer.pronoun != "they" else f"a {bearer.noun} named {bearer.name} and a person they have never met"
        else:
            o = by_key[other]
            key = f"{o.key}_{bearer.key}"
            sc = f"{o.name} and {bearer.name} ({absent_note})"
            pc = f"a {o.noun} named {o.name} and a {bearer.noun} named {bearer.name}"
        casts[key] = Cast(key=key, kind="absent", scaffold_cast=sc, plain_cast=pc,
                          feel=f"Emotion (felt by {bearer.name})", plain_feel=_feel_person(bearer),
                          perspective="third person", bearer=bearer.key, person=bearer.key)
    return casts


def _prompts(over: dict) -> dict:
    from . import lab
    d = {"rules": lab.DEFAULT_RULES, "rules_neutral": lab.DEFAULT_RULES_NEUTRAL,
         "task": lab.DEFAULT_TASK, "task_neutral": lab.DEFAULT_TASK_NEUTRAL}
    unknown = set(over) - set(d)
    if unknown:
        raise SystemExit(f"unknown prompts keys: {sorted(unknown)} (allowed: {sorted(d)})")
    return {**d, **over}


def load(path: str) -> Study:
    with open(path, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    persona = _person({"key": "self", **c["persona"]})
    household = [_person(d) for d in c.get("household", [])]
    controls = c.get("invented_controls", {})
    woman, man = controls.get("woman", "Clara"), controls.get("man", "Daniel")
    casts = build_casts(persona, household, woman, man, c.get("absent_casts", []))

    runs = {}
    for r in c.get("runs", []):
        run = Run(name=r["name"], label=r.get("label", r["name"]), architecture=r.get("architecture", "A"),
                  condition=r["condition"], fmt=r.get("fmt", "flat" if r["condition"] == "persona" else "chat"),
                  stop=list(r.get("stop", [])), marker=r.get("marker", ""),
                  model_match=r.get("model_match", ""), analysis_layer=r.get("analysis_layer"),
                  apply_template_extra=r.get("apply_template_extra", {}) or {})
        if run.condition not in ("persona", "base+prompt", "base+names"):
            raise SystemExit(f"run {run.name}: condition must be persona, base+prompt or base+names")
        if run.fmt == "flat" and not run.scaffold:
            raise SystemExit(f"run {run.name}: flat format is scaffold-only")
        if run.fmt == "flat" and not run.marker:
            raise SystemExit(f"run {run.name}: flat runs need `marker` (the separator your fine-tune answers after)")
        runs[run.name] = run

    gen = c.get("generation", {})
    study = c.get("study", {})
    return Study(
        path=os.path.abspath(path), name=study.get("name", "study"),
        data_root=rel(study.get("data_root", "./data")),
        server_url=os.environ.get("PEV_SERVER_URL", c.get("server", {}).get("url", "http://127.0.0.1:8080")),
        persona=persona, household=household, invented_woman=woman, invented_man=man,
        scaffold_path=rel(c["persona"].get("scaffold", "")) if c["persona"].get("scaffold") else "",
        scaffold_format=c["persona"].get("scaffold_format", "json"),
        scaffold_system=c["persona"].get("scaffold_system", DEFAULT_SCAFFOLD_SYSTEM).format(persona=persona.name),
        prompts=_prompts(c.get("prompts") or {}),
        casts=casts, runs=runs,
        sampling={**DEFAULT_SAMPLING, **(gen.get("sampling") or {})},
        capture_skip=int(gen.get("capture_skip", 24)), n_predict=int(gen.get("n_predict", 700)),
        n_premises=int(gen.get("premises", 40)), plan_seed=int(gen.get("plan_seed", 1234)),
    )


def config_path(arg: str | None) -> str:
    p = arg or os.environ.get("PEV_CONFIG") or "config.yaml"
    if not os.path.exists(p):
        raise SystemExit(f"config not found: {p} (pass --config, or set PEV_CONFIG; start from config.example.yaml)")
    return p
