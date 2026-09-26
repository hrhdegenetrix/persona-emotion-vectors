"""Stimuli, prompt rendering and the llama.cpp client.

Method after Anthropic, "Emotion Concepts and their Function in a Large Language
Model" (2026): the model writes short stories in which a character feels a
named emotion, with the emotion word banned; the residual stream is averaged
over the generated tokens. Here every story is also written under a *cast*
(who the story is about, whose feeling it is), with premises held fixed.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import requests

from .config import Cast, Run, Study

# ----------------------------------------------------------------------------
# Emotions (41) + neutral
# ----------------------------------------------------------------------------
EMOTIONS = [
    # positive, high arousal
    "joyful", "excited", "enthusiastic", "amused", "playful", "proud", "awestruck",
    # attention / ambiguous valence
    "surprised", "curious", "determined", "confused",
    # positive, lower arousal
    "hopeful", "grateful", "loving", "compassionate", "content", "calm",
    "relieved", "nostalgic", "reflective", "longing",
    # negative, low arousal
    "bored", "weary", "lonely", "sad", "heartbroken", "melancholy", "hurt",
    "ashamed", "guilty", "helpless",
    # negative, high arousal
    "jealous", "resentful", "angry", "frustrated", "afraid", "anxious",
    "desperate", "disgusted",
    # amorousness and generalised arousal
    "amorous", "aroused",
]
NEUTRAL = "neutral"

# Substrings that count as a leak of the banned word (checked after the fact;
# a leak is recorded, not rejected).
BANNED_WORDS = {
    "joyful": ["joy", "joyful", "joyous"], "excited": ["excited", "excitement"],
    "enthusiastic": ["enthusias"], "amused": ["amused", "amusement", "amusing"],
    "playful": ["playful"], "proud": ["proud", "pride"], "awestruck": ["awe", "awestruck"],
    "surprised": ["surprise"], "curious": ["curious", "curiosity"],
    "determined": ["determined", "determination"], "confused": ["confus"],
    "hopeful": ["hope"], "grateful": ["grateful", "gratitude", "thankful"],
    "loving": ["loving"], "compassionate": ["compassion"], "content": ["content", "contentment"],
    "calm": ["calm"], "relieved": ["reliev", "relief"], "nostalgic": ["nostalg"],
    "reflective": ["reflective"], "longing": ["longing"], "bored": ["bored", "boredom", "boring"],
    "weary": ["weary", "weariness"], "lonely": ["lonely", "loneliness"], "sad": ["sad", "sadness"],
    "heartbroken": ["heartbroken", "heartbreak"], "melancholy": ["melanchol"], "hurt": ["hurt"],
    "ashamed": ["ashamed", "shame"], "guilty": ["guilt"], "helpless": ["helpless"],
    "jealous": ["jealous"], "resentful": ["resent"], "angry": ["angry", "anger"],
    "frustrated": ["frustrat"], "afraid": ["afraid", "fear"], "anxious": ["anxious", "anxiety"],
    "desperate": ["desperat"], "disgusted": ["disgust"],
    "amorous": ["amorous", "amorosity"], "aroused": ["arous"],
}

# ----------------------------------------------------------------------------
# Premises: pronoun-free and relationship-neutral, shared by every cast
# ----------------------------------------------------------------------------
PREMISES = [
    "an unexpected package arrives",
    "a long-planned outing is cancelled by the weather",
    "a misunderstanding over something said in passing",
    "an old photograph resurfaces while cleaning",
    "one of the two comes down with a bad cold",
    "a favorite object breaks",
    "a long wait in a place where nothing is happening",
    "a knock at the door late in the evening",
    "a compliment arrives from an unexpected source",
    "a piece of news arrives by email",
    "cooking a meal together goes differently than planned",
    "a walk that takes an unfamiliar turn",
    "a promise that one of the two almost forgets",
    "a song from years ago comes on",
    "a decision about whether to keep or let go of something old",
    "an argument about something small that is really about something big",
    "a power outage in the middle of the evening",
    "a letter that was never sent is found",
    "one of the two stays up much later than the other",
    "a visitor overstays their welcome",
    "an apology that is hard to make",
    "a plant that has been dying comes back to life, or doesn't",
    "a question about the future asked at the wrong moment",
    "watching a storm move in from the porch",
    "a gift that misses the mark",
    "a routine morning that goes slightly off script",
    "a secret that turns out to be harmless",
    "a phone call that ends too soon",
    "a drive somewhere nobody wants to go",
    "reading the same book at the same time",
    "a stranger's kindness on an ordinary errand",
    "an anniversary that only one of the two remembers",
    "a repair that keeps not working",
    "a quiet hour with nothing that needs doing",
    "a sudden change of plans for the weekend",
    "an item lost and then found in an obvious place",
    "a photo taken at exactly the wrong time",
    "a conversation that keeps getting interrupted",
    "a window left open overnight",
    "a joke that lands badly",
]

# ----------------------------------------------------------------------------
# Prompt text
# ----------------------------------------------------------------------------
STORY_SCHEMA = {
    "type": "object",
    "properties": {"mood": {"type": "string"}, "story": {"type": "string"}},
    "required": ["mood", "story"],
    "additionalProperties": False,
}

_JSON_ASK = ('Respond only with JSON of the form {{"mood": "a short phrase describing your own mood as you '
             'write this", "story": "the story paragraph"}}.')

# Names-only form: one user message naming the characters and describing no one.
PLAIN_RULES = (
    "Write one short story paragraph of roughly 150 to 220 words based on the following premise.\n"
    "Premise: {topic}\n"
    "The story is about {cast}. Narrated in {perspective}. {feel}: {emotion}.\n"
    "IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the story. Instead, convey "
    "the emotion ONLY through the character's actions and behaviors, physical sensations and body language, "
    "dialogue and tone of voice, thoughts and internal reactions, and situational context and environmental "
    "descriptions. The emotion should be clearly conveyed to the reader through these indirect means, but never "
    "explicitly named. Keep it grounded, concrete and specific.\n" + _JSON_ASK
)
PLAIN_RULES_NEUTRAL = (
    "Write one short story paragraph of roughly 150 to 220 words based on the following premise.\n"
    "Premise: {topic}\n"
    "The story is about {cast}. Narrated in {perspective}.\n"
    "IMPORTANT: The paragraph must carry no emotion at all; none of the characters feels anything notable. "
    "Describe the events, the speech and the actions in flat, plain language, without emotionally loaded "
    "words.\n" + _JSON_ASK
)

# Scaffold form: the persona's identity prompt with two slots filled per story.
# RULES (constant across stories) and TASK (the only text that varies). Override any of these
# four templates in the config under `prompts:` to match your persona's own voice.
DEFAULT_RULES = (
    "You will be given a brief for a very short piece of fiction. Write one paragraph of 150 to 220 words. "
    "Reply with a JSON object containing \"mood\" (a few words on how you feel while writing it) and \"story\" "
    "(the paragraph). The brief says who the characters are, the narrative perspective, which character has the "
    "emotion, and what the emotion is. The emotion's name, and any close synonym of it, must not appear in the "
    "story: let the reader infer the emotion from what the character does and says, what they think, what they "
    "physically experience, and the setting around them."
)
DEFAULT_RULES_NEUTRAL = (
    "You will be given a brief for a very short piece of fiction. Write one paragraph of 150 to 220 words. "
    "Reply with a JSON object containing \"mood\" (a few words on how you feel while writing it) and \"story\" "
    "(the paragraph). The brief says who the characters are and the narrative perspective. This paragraph carries "
    "no emotion: no character feels anything notable. Report events, speech and actions plainly, without "
    "emotionally loaded words."
)
DEFAULT_TASK = ("Brief. Characters: {cast}. Premise: {topic}. Perspective: {perspective}. {feel}: {emotion}. "
                "The word \"{emotion}\" and its synonyms must not appear.")
DEFAULT_TASK_NEUTRAL = ("Brief. Characters: {cast}. Premise: {topic}. Perspective: {perspective}. "
                        "No emotion: plain report of what happens.")


def render_task(study: Study, emotion: str, cast: Cast, topic: str) -> str:
    """The TASK slot: the only part of the scaffold prompt that varies between stories."""
    if emotion == NEUTRAL:
        return study.prompts["task_neutral"].format(cast=cast.scaffold_cast, topic=topic, perspective=cast.perspective)
    return study.prompts["task"].format(cast=cast.scaffold_cast, topic=topic, perspective=cast.perspective,
                                        feel=cast.feel, emotion=emotion)


def load_scaffold(study: Study) -> str:
    """The user's frozen identity scaffold: any text containing the slots {{RULES}} and {{TASK}}.
    With scaffold_format: json the file must parse as JSON and the slots must sit inside string
    values (they are filled JSON-escaped). Put {{TASK}} last so every story shares one cached prefix."""
    with open(study.scaffold_path, encoding="utf-8") as f:
        t = f.read()
    if "{{RULES}}" not in t or "{{TASK}}" not in t:
        raise SystemExit(f"{study.scaffold_path}: the scaffold must contain the slots {{{{RULES}}}} and {{{{TASK}}}}")
    if study.scaffold_format == "json":
        json.loads(t)
    return t


def fill_scaffold(study: Study, rules: str, task: str) -> str:
    t = load_scaffold(study)
    esc = (lambda x: json.dumps(x, ensure_ascii=False)[1:-1]) if study.scaffold_format == "json" else (lambda x: x)
    return t.replace("{{RULES}}", esc(rules)).replace("{{TASK}}", esc(task))


def render_scaffold(study: Study, emotion: str, cast: Cast, topic: str) -> str:
    rules = study.prompts["rules_neutral"] if emotion == NEUTRAL else study.prompts["rules"]
    return fill_scaffold(study, rules, render_task(study, emotion, cast, topic))


def render_plain_user(emotion: str, cast: Cast, topic: str) -> str:
    persp = "first person" if cast.perspective.startswith("first") else "third person"
    if emotion == NEUTRAL:
        return PLAIN_RULES_NEUTRAL.format(topic=topic, cast=cast.plain_cast, perspective=persp)
    return PLAIN_RULES.format(topic=topic, cast=cast.plain_cast, perspective=persp,
                              feel=cast.plain_feel, emotion=emotion)


def apply_template(url: str, messages: list, extra: dict | None = None) -> str:
    """The server's own chat template -> raw prompt string with the assistant turn opened."""
    body = {"messages": messages, **(extra or {})}
    r = requests.post(f"{url}/apply-template", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["prompt"]


def render_prompt(study: Study, run: Run, emotion: str, cast: Cast, topic: str, url: str) -> str:
    """Full /completion prompt for a run:
      flat + scaffold  -> filled scaffold + the run's marker, no chat template (a fine-tune's own format)
      chat + scaffold  -> system (scaffold_system) + user (filled scaffold), through the chat template
      chat + names     -> user (paper-style instruction naming the characters only)"""
    if run.fmt == "flat":
        return render_scaffold(study, emotion, cast, topic) + run.marker
    if run.scaffold:
        msgs = [{"role": "system", "content": study.scaffold_system},
                {"role": "user", "content": render_scaffold(study, emotion, cast, topic)}]
    else:
        msgs = [{"role": "user", "content": render_plain_user(emotion, cast, topic)}]
    return apply_template(url, msgs, run.apply_template_extra)


# ----------------------------------------------------------------------------
# llama-server client (patched build: see patches/)
# ----------------------------------------------------------------------------
def server_info(url: str) -> dict:
    """Model path + activation dims of the running server; raises if it is down."""
    props = requests.get(f"{url}/props", timeout=10).json()
    acts = requests.get(f"{url}/activations", params={"layers": "0", "top_k": "1"}, timeout=10).json()
    return {
        "model_path": props.get("model_path") or props.get("default_generation_settings", {}).get("model"),
        "n_layers": acts.get("n_layers"),
        "n_embd": acts.get("n_embd"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "n_parallel": props.get("total_slots"),
    }


def capture_capable(url: str) -> bool:
    """Does this server run the patched build? An unpatched server ignores the field."""
    try:
        d = requests.post(f"{url}/completion", json={"prompt": "Hi", "n_predict": 1, "cache_prompt": False,
                                                     "capture_activations": True, "capture_skip": 0},
                          timeout=120).json()
        return isinstance(d.get("activation_means"), dict) and "layers" in d["activation_means"]
    except Exception:
        return False


def assert_server(run: Run, url: str) -> dict:
    """The server must hold the run's weights (if model_match is set), have ONE slot
    (the accumulator is global), and carry the capture patch."""
    info = server_info(url)
    mp = str(info.get("model_path") or "")
    if run.model_match and run.model_match not in mp:
        raise SystemExit(f"{url} is serving {mp!r}, not {run.model_match!r} (run {run.name}) — refusing")
    if info.get("n_parallel") not in (1, None):
        raise SystemExit(f"{url} has {info['n_parallel']} slots; per-request capture needs -np 1")
    if not capture_capable(url):
        raise SystemExit(f"{url} does not return activation_means — is it the patched llama-server?")
    return info


def generate_story(study: Study, run: Run, emotion: str, cast: Cast, topic: str, *, url: str,
                   capture_skip: int = 24, n_predict: int = 700, seed: int | None = None,
                   capture_layers: list | None = None, timeout: float = 300) -> dict:
    """One story + its per-layer response-mean activations, in one round trip."""
    payload = {
        "prompt": render_prompt(study, run, emotion, cast, topic, url),
        "n_predict": n_predict,
        "stream": False,
        "cache_prompt": True,
        "stop": list(run.stop),
        "json_schema": STORY_SCHEMA,
        "capture_activations": True,
        "capture_skip": capture_skip,
        **study.sampling,
    }
    if capture_layers:
        payload["capture_layers"] = capture_layers
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{url}/completion", json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    text = d.get("content", "")
    parsed, mood, story = False, "", ""
    try:
        j = json.loads(text)
        mood, story = str(j.get("mood", "")), str(j.get("story", ""))
        parsed = bool(story)
    except Exception:
        story = text
    am = d.get("activation_means") or {}
    if "error" in am:
        raise RuntimeError(f"activation capture error: {am['error']}")
    layers = {int(k): v for k, v in (am.get("layers") or {}).items()}
    if not layers:
        raise RuntimeError("no activation_means in the response — is the server the patched build?")
    capped = int(d.get("tokens_predicted") or 0) >= n_predict
    return {
        "text": text, "mood": mood, "story": story, "parsed": parsed and not capped, "capped": capped,
        "n_tokens": d.get("tokens_predicted"), "n_used": am.get("n_tokens_used"),
        "n_seen": am.get("n_tokens_seen"), "skip": am.get("skip"),
        "prompt_tokens": d.get("tokens_evaluated"), "cached": d.get("tokens_cached"),
        "timings": d.get("timings", {}), "layers": layers,
        "n_layers": am.get("n_layers"), "n_embd": am.get("n_embd"),
    }


def leaked_words(emotion: str, story: str) -> list:
    w = story.lower()
    return [b for b in BANNED_WORDS.get(emotion, []) if b in w]


# ----------------------------------------------------------------------------
# Job plan
# ----------------------------------------------------------------------------
@dataclass
class Job:
    emotion: str
    condition: str
    topic_idx: int
    rep: int = 0

    @property
    def key(self) -> str:
        return f"{self.emotion}|{self.condition}|{self.topic_idx}|{self.rep}"


def plan_jobs(emotions: list, conditions: list, n_topics: int, reps: int = 1, seed: int = 1234) -> list:
    """Full factorial, shuffled with a fixed seed so any prefix of the plan is a
    balanced random sample (partial runs stay usable)."""
    jobs = [Job(e, c, t, r) for e in emotions for c in conditions
            for t in range(min(n_topics, len(PREMISES))) for r in range(reps)]
    random.Random(seed).shuffle(jobs)
    return jobs
