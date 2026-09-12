import json
import os
import re
import sys

from dotenv import load_dotenv

# Make `modules/` importable regardless of where this file is run from.
# chatbot/response_engine.py -> project root is one level up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    from modules.soil_adjustment import compute_nutrient_gap, get_supported_crops
    from modules.fertilizer_recommendation import recommend_fertilizer, rec_label
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    rec_label = None

try:
    from modules.yield_model import predict_yield, load_model as load_yield_model
    YIELD_MODULE_AVAILABLE = True
except ImportError:
    YIELD_MODULE_AVAILABLE = False

from language import CROP_ALIASES, resolve_language
from i18n import crop_name, fert_phrase, format_crop_list, msg, nutrient_name, status_name


def _get_supported_crops(report):
    """Prefer the report's own list; fall back to the live module."""
    crops = report.get("supported_crops")
    if crops:
        return crops
    if MODULES_AVAILABLE:
        return get_supported_crops()
    return []


def _extract_mentioned_crop(user_message, supported_crops):
    """
    Look for any supported crop name mentioned in the user's message
    (English or Hindi/Marathi aliases, whole-word, case-insensitive).
    """
    if not user_message:
        return None

    msg_lower = user_message.lower()
    supported = set(supported_crops)

    alias_pairs = sorted(CROP_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, crop in alias_pairs:
        if crop not in supported:
            continue
        pattern = r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"
        if re.search(pattern, msg_lower, flags=re.IGNORECASE):
            return crop

    for crop in supported_crops:
        pattern = r"(?<!\w)" + re.escape(crop.lower()) + r"(?!\w)"
        if re.search(pattern, msg_lower, flags=re.IGNORECASE):
            return crop

    return None


def _rec_label(rec):
    if rec_label is not None:
        return rec_label(rec)
    if rec.get("condition") == "High":
        return rec.get("action") or "No application"
    return rec.get("fertilizer") or "Unknown"


def _resolve_target_crop(user_message, report, chat_context=None):
    """
    Figure out which crop the user is actually asking about.

    Returns (target_crop, is_override, is_unsupported_mention):
      target_crop          — crop to compute/report on
      is_override          — True if the user asked about a crop other
                              than report['chosen_crop']
      is_unsupported_mention — True if the user named a crop that isn't
                              in the supported list at all (e.g. Gram)
    """
    chosen_crop = report.get("chosen_crop")
    supported_crops = _get_supported_crops(report)

    mentioned = _extract_mentioned_crop(user_message, supported_crops)

    if mentioned is None:
        if chat_context and chat_context.active_crop:
            target = chat_context.active_crop
            return target, target != chosen_crop, False
        return chosen_crop, False, False

    if chat_context:
        chat_context.set_active_crop(mentioned)

    if mentioned == chosen_crop:
        return chosen_crop, False, False

    return mentioned, True, False


def _get_top_crop(report):
    """The model's actual #1 recommended crop (not the dropdown target)."""
    top_crop = report.get("top_crop")
    if top_crop:
        return top_crop
    top_crops = report.get("top_5_crops")
    if top_crops:
        return top_crops[0]["crop"]
    return report.get("chosen_crop")


def _build_override_prefix(report, target_crop, needs_recompute, lang):
    """
    Show a note only when the crop being discussed differs from the
    model's actual top recommendation — not merely from the dropdown
    target crop (which cached report data is computed for).
    """
    if not needs_recompute:
        return ""
    top_crop = _get_top_crop(report)
    if not top_crop or target_crop == top_crop:
        return ""
    return _override_prefix(target_crop, top_crop, lang)


def _override_prefix(target_crop, top_crop, lang):
    return msg(
        lang,
        "OVERRIDE_PREFIX",
        chosen=crop_name(top_crop, lang),
        crop=crop_name(target_crop, lang),
    )


def _yield_interp_key(yield_info):
    pred = yield_info.get("predicted_yield_kg_ha")
    if not isinstance(pred, (int, float)):
        return None
    if pred < 1000:
        return "YIELD_LOW"
    if pred < 3000:
        return "YIELD_MOD"
    if pred < 15000:
        return "YIELD_GOOD"
    return "YIELD_HIGH"


def _report_facts(report):
    """Subset of the prediction report the LLM is allowed to cite."""
    fertilizer = report.get("fertilizer") or {}
    return {
        "chosen_crop": report.get("chosen_crop"),
        "top_crop": report.get("top_crop"),
        "top_5_crops": report.get("top_5_crops"),
        "soil_input": report.get("soil_input"),
        "soil_adjustment": report.get("soil_adjustment"),
        "fertilizer": {
            "action_needed": fertilizer.get("action_needed"),
            "generic_recs": fertilizer.get("generic_recs"),
            "crop_primary": fertilizer.get("crop_primary"),
        },
        "yield_prediction": report.get("yield_prediction"),
    }


def generate_explanation(report):
    """
    Ask Groq to explain the recommendation in 2–3 English sentences.

    Returns the explanation string, or None if the API key is missing
    or the call fails (timeout, rate limit, network error, etc.).
    """
    if not report:
        return None

    api_key = os.getenv("GROQ_API_KEY")
    print(f"DEBUG: BASE_DIR={BASE_DIR}, env_path_exists={os.path.exists(os.path.join(BASE_DIR, '.env'))}, api_key_found={bool(api_key)}")
    if not api_key:
        return None

    facts = _report_facts(report)
    try:
        payload = json.dumps(facts, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None

    prompt = (
        "You explain a Smart Agri crop/soil/fertilizer/yield recommendation "
        "to a farmer in 2-3 short sentences of English.\n"
        "ONLY reference numbers, crop names, fertilizer names, statuses, "
        "and other values that appear in the JSON report below.\n"
        "Do not invent, estimate, round into new figures, or add any number "
        "that is not present in the report.\n"
        "Do not mention tools, models, APIs, or that you are an AI.\n\n"
        f"Report JSON:\n{payload}"
    )

    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=20.0)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write brief farmer-facing explanations. "
                        "You may only use facts supplied in the user message."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            reasoning_effort="low",
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"EXPLANATION RESULT: {repr(text)}")
        return text or None
    except Exception as e:
        print(f"EXPLANATION ERROR: {e}")
        return None


def generate_response(intent, report, user_message="", chat_context=None, lang=None):
    """
    Generate a response using the predicted intent, the Smart Agri
    report, and (when relevant) the crop the user actually named in
    their message — which may differ from report['chosen_crop'].
    Replies in English, Hindi, or Marathi.
    """
    lang = resolve_language(user_message, preferred=lang)

    if intent == "GREETING":
        return msg(lang, "GREETING")

    if intent == "HELP":
        return msg(lang, "HELP")

    if intent == "SYSTEM_OVERVIEW":
        return msg(lang, "SYSTEM_OVERVIEW")

    if intent == "HOW_TO_USE":
        return msg(lang, "HOW_TO_USE")

    if intent == "DASHBOARD":
        return msg(lang, "DASHBOARD")

    if intent == "ML_MODEL_EXPLANATION":
        return msg(lang, "ML_MODEL_EXPLANATION")

    if intent == "SUPPORTED_CROPS":
        crops = report.get("supported_crops", [])
        if not crops:
            return msg(lang, "SUPPORTED_CROPS_EMPTY")
        return msg(lang, "SUPPORTED_CROPS", crops=format_crop_list(crops, lang))

    if intent == "CROP_RECOMMENDATION":
        top_crops = report.get("top_5_crops")
        crop = report.get("top_crop") or (top_crops[0]["crop"] if top_crops else report["chosen_crop"])
        return msg(lang, "CROP_RECOMMENDATION", crop=crop_name(crop, lang))

    if intent == "CROP_RECOMMENDATION_EXPLANATION":
        top_crops = report.get("top_5_crops")
        crop = report.get("top_crop") or (top_crops[0]["crop"] if top_crops else report["chosen_crop"])
        if not top_crops:
            return msg(
                lang,
                "CROP_RECOMMENDATION_EXPLANATION_NO_PROBS",
                crop=crop_name(crop, lang),
            )
        probability = top_crops[0]["probability"] * 100
        return msg(
            lang,
            "CROP_RECOMMENDATION_EXPLANATION",
            crop=crop_name(crop, lang),
            probability=probability,
        )

    if intent == "CROP_PROBABILITY":
        recommendations = report.get("top_5_crops")
        if not recommendations:
            return msg(lang, "CROP_PROBABILITY_EMPTY")
        lines = [msg(lang, "CROP_PROBABILITY_HEADER")]
        for item in recommendations:
            lines.append(
                msg(
                    lang,
                    "CROP_PROBABILITY_LINE",
                    crop=crop_name(item["crop"], lang),
                    probability=item["probability"] * 100,
                )
            )
        return "\n".join(lines)

    if intent == "CROP_SOIL_MISMATCH":
        chosen_crop = report.get("chosen_crop")
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if not MODULES_AVAILABLE:
            gap_result = report.get("soil_adjustment", {})
        else:
            try:
                gap_result = compute_nutrient_gap(target_crop, soil_input)
            except (ValueError, KeyError):
                gap_result = None

        if not gap_result:
            return msg(
                lang,
                "CROP_SOIL_UNSUPPORTED",
                crop=crop_name(target_crop, lang),
                supported=format_crop_list(_get_supported_crops(report), lang),
            )

        mismatches = [
            msg(
                lang,
                "SOIL_MISMATCH_LINE",
                nutrient=nutrient_name(nutrient, lang),
                status=status_name(info["status"], lang),
                current=info["current"],
                ideal=info["ideal"],
            )
            for nutrient, info in gap_result.items()
            if info["status"] != "Optimal"
        ]

        prefix = _build_override_prefix(report, target_crop, is_override, lang)

        if not mismatches:
            return msg(
                lang,
                "CROP_SOIL_MATCH",
                prefix=prefix,
                crop=crop_name(target_crop, lang),
            )

        return msg(
            lang,
            "CROP_SOIL_MISMATCH",
            prefix=prefix,
            crop=crop_name(target_crop, lang),
            lines="\n".join(mismatches),
        )

    if intent == "SOIL_ANALYSIS":
        soil = report["soil_input"]
        return msg(
            lang,
            "SOIL_ANALYSIS",
            n=soil["Nitrogen"],
            p=soil["Phosphorus"],
            k=soil["Potassium"],
            ph=soil["pH"],
            temp=soil["Temperature"],
            rain=soil["Rainfall"],
        )

    nutrient_intents = {
        "SOIL_NITROGEN": "Nitrogen",
        "SOIL_PHOSPHORUS": "Phosphorus",
        "SOIL_POTASSIUM": "Potassium",
        "SOIL_PH": "pH",
    }

    if intent in nutrient_intents:
        nutrient = nutrient_intents[intent]
        soil_input = report.get("soil_input", {})
        chosen_crop = report.get("chosen_crop")
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and MODULES_AVAILABLE:
            try:
                soil_adjustment = compute_nutrient_gap(target_crop, soil_input)
            except (ValueError, KeyError):
                soil_adjustment = {}
        else:
            soil_adjustment = report.get("soil_adjustment", {})

        current_value = soil_input.get(nutrient)
        unit = "" if nutrient == "pH" else " kg/ha"

        if nutrient not in soil_adjustment:
            return msg(
                lang,
                "NUTRIENT_NO_ADJUSTMENT",
                nutrient=nutrient_name(nutrient, lang),
                value=current_value,
                unit=unit,
                crop=crop_name(target_crop, lang),
            )

        info = soil_adjustment[nutrient]
        prefix = _build_override_prefix(report, target_crop, is_override, lang)
        return msg(
            lang,
            "NUTRIENT_STATUS",
            prefix=prefix,
            nutrient=nutrient_name(nutrient, lang),
            current=info["current"],
            unit=unit,
            crop=crop_name(target_crop, lang),
            ideal=info["ideal"],
            status=status_name(info["status"], lang),
        )

    if intent == "SOIL_STATUS_EXPLANATION":
        return msg(lang, "SOIL_STATUS_EXPLANATION")

    if intent == "FERTILIZER_RECOMMENDATION":
        chosen_crop = report.get("chosen_crop")
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and MODULES_AVAILABLE:
            try:
                fertilizer = recommend_fertilizer(target_crop, soil_input)
            except (ValueError, KeyError):
                fertilizer = None
        else:
            fertilizer = report.get("fertilizer")

        if not fertilizer or not fertilizer.get("gap_analysis"):
            return msg(
                lang,
                "FERTILIZER_UNSUPPORTED",
                crop=crop_name(target_crop, lang),
                supported=format_crop_list(_get_supported_crops(report), lang),
            )

        prefix = _build_override_prefix(report, target_crop, is_override, lang)

        if not fertilizer["action_needed"]:
            return msg(
                lang,
                "FERTILIZER_BALANCED",
                prefix=prefix,
                crop=crop_name(target_crop, lang),
            )

        lines = [msg(lang, "FERTILIZER_BASED", prefix=prefix)]
        for rec in fertilizer["generic_recs"]:
            if rec["condition"] in ("Low", "High"):
                lines.append(
                    msg(
                        lang,
                        "FERTILIZER_CONDITION_LINE",
                        nutrient=nutrient_name(rec["nutrient"], lang),
                        status=status_name(rec["condition"], lang),
                        label=_rec_label(rec),
                    )
                )

        if fertilizer["crop_primary"]:
            lines.append(
                msg(
                    lang,
                    "FERTILIZER_PRIMARY",
                    crop=crop_name(target_crop, lang),
                    fertilizer=fertilizer["crop_primary"]["fertilizer"],
                )
            )
        return "\n".join(lines)

    if intent == "FERTILIZER_DOSAGE":
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and MODULES_AVAILABLE:
            try:
                fertilizer = recommend_fertilizer(target_crop, soil_input)
            except (ValueError, KeyError):
                fertilizer = None
        else:
            fertilizer = report.get("fertilizer")

        if not fertilizer:
            return msg(
                lang,
                "FERTILIZER_DOSAGE_UNSUPPORTED",
                crop=crop_name(target_crop, lang),
            )

        recommendations = fertilizer.get("generic_recs", [])
        prefix = _build_override_prefix(report, target_crop, is_override, lang)

        if not recommendations:
            return msg(
                lang,
                "FERTILIZER_DOSAGE_NONE",
                prefix=prefix,
                crop=crop_name(target_crop, lang),
            )

        lines = [msg(lang, "FERTILIZER_DOSAGE_HEADER", prefix=prefix)]
        for rec in recommendations:
            lines.append(
                msg(
                    lang,
                    "FERTILIZER_DOSAGE_LINE",
                    label=_rec_label(rec),
                    dose=fert_phrase(rec["dose_note"], lang),
                )
            )
        return "\n".join(lines)

    if intent == "WHY_FERTILIZER_NEEDED":
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and MODULES_AVAILABLE:
            try:
                fertilizer = recommend_fertilizer(target_crop, soil_input)
            except (ValueError, KeyError):
                fertilizer = None
        else:
            fertilizer = report.get("fertilizer")

        if not fertilizer:
            return msg(
                lang,
                "WHY_FERT_UNSUPPORTED",
                crop=crop_name(target_crop, lang),
            )

        recommendations = fertilizer.get("generic_recs", [])
        prefix = _build_override_prefix(report, target_crop, is_override, lang)

        if not recommendations:
            return msg(
                lang,
                "WHY_FERT_NONE",
                prefix=prefix,
                crop=crop_name(target_crop, lang),
            )

        lines = [msg(lang, "WHY_FERT_HEADER", prefix=prefix)]
        for rec in recommendations:
            lines.append(
                msg(
                    lang,
                    "WHY_FERT_LINE",
                    label=_rec_label(rec),
                    reason=fert_phrase(rec["reason"], lang),
                )
            )
        return "\n".join(lines)

    if intent == "FERTILIZER_EXPLANATION":
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and MODULES_AVAILABLE:
            try:
                fertilizer = recommend_fertilizer(target_crop, soil_input)
            except (ValueError, KeyError):
                fertilizer = None
        else:
            fertilizer = report.get("fertilizer")

        if not fertilizer:
            return msg(
                lang,
                "WHY_FERT_UNSUPPORTED",
                crop=crop_name(target_crop, lang),
            )

        recommendations = fertilizer.get("generic_recs", [])
        prefix = _build_override_prefix(report, target_crop, is_override, lang)

        if not recommendations:
            return msg(
                lang,
                "FERT_EXPLAIN_NONE",
                prefix=prefix,
                crop=crop_name(target_crop, lang),
            )

        lines = [msg(lang, "FERT_EXPLAIN_HEADER", prefix=prefix)]
        for rec in recommendations:
            label = _rec_label(rec)
            if rec["condition"] == "Low":
                lines.append(
                    msg(
                        lang,
                        "FERT_EXPLAIN_LOW",
                        label=label,
                        npk=rec["npk_ratio"],
                        reason=fert_phrase(rec["reason"], lang),
                    )
                )
            else:
                lines.append(
                    msg(
                        lang,
                        "FERT_EXPLAIN_HIGH",
                        label=label,
                        reason=fert_phrase(rec["reason"], lang),
                    )
                )
        return "\n".join(lines)

    if intent == "YIELD_PREDICTION":
        chosen_crop = report.get("chosen_crop")
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and YIELD_MODULE_AVAILABLE:
            try:
                yield_model = load_yield_model()
                yield_info = predict_yield(soil_input, model=yield_model, crop=target_crop)
            except FileNotFoundError:
                yield_info = None
        else:
            yield_info = report.get("yield_prediction")

        if not yield_info or not isinstance(
            yield_info.get("predicted_yield_kg_ha"), (int, float)
        ):
            return msg(lang, "YIELD_NONE", crop=crop_name(target_crop, lang))

        prefix = _build_override_prefix(report, target_crop, is_override, lang)
        return msg(
            lang,
            "YIELD_PREDICTION",
            prefix=prefix,
            crop=crop_name(target_crop, lang),
            yield_kg=yield_info["predicted_yield_kg_ha"],
        )

    if intent == "YIELD_EXPLANATION":
        chosen_crop = report.get("chosen_crop")
        soil_input = report.get("soil_input", {})
        target_crop, is_override, _ = _resolve_target_crop(user_message, report, chat_context)

        if is_override and YIELD_MODULE_AVAILABLE:
            try:
                yield_model = load_yield_model()
                yield_info = predict_yield(soil_input, model=yield_model, crop=target_crop)
            except FileNotFoundError:
                yield_info = None
        else:
            yield_info = report.get("yield_prediction")

        interp_key = _yield_interp_key(yield_info) if yield_info else None
        if not yield_info or not interp_key:
            return msg(lang, "YIELD_INTERP_NONE", crop=crop_name(target_crop, lang))

        prefix = _build_override_prefix(report, target_crop, is_override, lang)
        return prefix + msg(lang, interp_key)

    if intent == "YIELD_FACTORS":
        return msg(lang, "YIELD_FACTORS")

    if intent == "YIELD_STATUS_EXPLANATION":
        return msg(lang, "YIELD_STATUS_EXPLANATION")

    if intent == "OUT_OF_SCOPE":
        return msg(lang, "OUT_OF_SCOPE")

    if intent == "UNKNOWN":
        return msg(lang, "UNKNOWN")

    return msg(lang, "FALLBACK")