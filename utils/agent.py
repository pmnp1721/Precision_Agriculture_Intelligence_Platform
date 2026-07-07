import json
import os

from .crop_model import recommend_crop
from .disease_model import load_disease_artifacts
from .irrigation_model import predict_irrigation
from .yield_model import forecast_crop_yield


def _gemini_model(api_key):
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except Exception:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-flash-latest")


def route_query(query, has_image=False):
    text = query.lower()
    routes = []
    if any(word in text for word in ["crop", "grow", "recommend", "suitable", "soil"]):
        routes.append("crop")
    if any(word in text for word in ["irrigation", "water", "moisture", "rain", "dry"]):
        routes.append("irrigation")
    if any(word in text for word in ["yield", "harvest", "production", "forecast"]):
        routes.append("yield")
    if any(word in text for word in ["disease", "leaf", "blight", "rust", "spot", "scorch"]):
        routes.append("disease")
    if has_image and "disease" not in routes:
        routes.append("disease")
    if not routes:
        routes = ["general"]
    return routes


def _fallback_answer(query, outputs):
    lines = ["🌾 Farm Advisory Report:"]
    if outputs.get("crop"):
        crop = outputs["crop"]["recommended_crop"]
        lines.append(f"- Recommended crop: {crop}.")
    if outputs.get("irrigation"):
        need = outputs["irrigation"]["irrigation_need"]
        lines.append(f"- Irrigation need: {need}.")
    if outputs.get("yield"):
        result = outputs["yield"]
        lines.append(
            f"- Forecast yield: {result['predicted_yield_hg_per_ha']:,} "
            f"{result['unit']} with {result['confidence_tier']} confidence."
        )
    if outputs.get("disease"):
        disease = outputs["disease"]
        if disease.get("image"):
            image = disease["image"]
            lines.append(
                f"- Leaf image received: {image['name']} "
                f"({image['width']} x {image['height']} px)."
            )
        lines.append(f"- Disease module: {disease['message']}")
        if disease.get("supported_classes"):
            lines.append(
                "- Supported disease classes: "
                + ", ".join(disease["supported_classes"][:9])
                + "."
            )
        lines.append(
            "- Immediate field action: isolate heavily affected leaves, compare visible symptoms "
            "against the supported classes, avoid overhead watering, and consult a local agronomist "
            "before chemical treatment."
        )
    if len(lines) == 1:
        lines.append(
            "Ask about crop choice, irrigation, yield, or disease and provide the farm context on this page."
        )
    lines.append(f"\nFarmer query: {query}")
    return "\n".join(lines)


def run_farming_agent(
    query,
    context,
    api_key=None,
    image_context=None,
    image_result=None,
    image_payload=None,
):
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    routes = route_query(query, has_image=bool(image_context))
    outputs = {}
    errors = []

    try:
        if "crop" in routes:
            outputs["crop"] = recommend_crop(context["crop"])
    except Exception as exc:
        errors.append(f"Crop recommendation failed: {exc}")

    try:
        if "irrigation" in routes:
            outputs["irrigation"] = predict_irrigation(context["irrigation"])
    except Exception as exc:
        errors.append(f"Irrigation prediction failed: {exc}")

    try:
        if "yield" in routes:
            outputs["yield"] = forecast_crop_yield(context["yield"])
    except Exception as exc:
        errors.append(f"Yield forecasting failed: {exc}")

    if "disease" in routes:
        supported_classes = []
        try:
            artifacts = load_disease_artifacts()
            class_col = artifacts["class_summary"].columns[0]
            supported_classes = artifacts["class_summary"][class_col].astype(str).tolist()
        except Exception as exc:
            errors.append(f"Disease artifact lookup failed: {exc}")

        outputs["disease"] = {
            "available": False,
            "message": (
                image_result["message"]
                if image_result
                else "Live image classification is currently unavailable because the trained disease model artifact is not present. Disease analytics and class references are still available on the Disease Detection page."
            ),
            "image": image_context,
            "supported_classes": supported_classes,
        }

    model = _gemini_model(api_key)
    if model:
        prompt = f"""
You are a practical AI farming advisor for farmers. Answer only the farmer's question.
Do not add extra sections, soil analysis, climate analysis, or general recommendations
unless they are directly needed to answer the question.
Use the local tool outputs as trusted evidence. Mention confidence and warnings only when relevant.
If an image is attached, describe it only as visual guidance, and do not claim the local disease
model classified it unless the tool output says live disease prediction is available.
Keep the tone helpful and constructive. If the disease model is not available, state that
visual inspection guidance is provided and recommend consulting a local agronomist.
Respond in 3 short bullet points or fewer. Do not use headings unless the user asked for them.
Do not mention tool traces, JSON, internal routes, or implementation details.

Farmer question:
{query}

Tool routes used:
{routes}

Tool outputs JSON:
{json.dumps(outputs, indent=2)}

Errors:
{errors}
"""
        try:
            content = [prompt, image_payload] if image_payload is not None else prompt
            response = model.generate_content(content)
            answer = response.text
        except Exception as exc:
            errors.append(f"Gemini response failed: {exc}")
            answer = _fallback_answer(query, outputs)
    else:
        answer = _fallback_answer(query, outputs)

    return {
        "routes": routes,
        "tool_outputs": outputs,
        "errors": errors,
        "answer": answer,
        "llm_used": bool(model),
    }
