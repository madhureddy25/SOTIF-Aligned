import json

import httpx
import numpy as np
import pytest

from sotif_miner.schemas import RoiAudit, SceneAudit, TriggerCardDraft
from sotif_miner.vlm.base import VLMError, inline_schema, parse_or_none
from sotif_miner.vlm.ollama import OllamaBackend
from sotif_miner.vlm.openai_compat import OpenAICompatBackend
from sotif_miner.vlm.scripted import ScriptedBackend

IMG = np.zeros((40, 60, 3), np.uint8)
ROI_JSON = '{"object_class": "pedestrian", "visibility": 2, "occluder_present": false, "notes": "none"}'


def test_inline_schema_has_no_refs():
    js = inline_schema(TriggerCardDraft)
    assert "$defs" not in json.dumps(js) and "$ref" not in json.dumps(js)
    assert set(js["properties"]["triggering_condition"]["enum"]) >= {"glare", "rain", "none"}


def test_parse_tolerates_code_fences():
    assert parse_or_none(RoiAudit, "```json\n" + ROI_JSON + "\n```").visibility == 2
    assert parse_or_none(RoiAudit, "Sure! " + ROI_JSON) is not None
    assert parse_or_none(RoiAudit, '{"object_class": "x"}') is None


def test_ollama_request_shape_and_retry():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        text = "not json" if len(seen) == 1 else ROI_JSON
        return httpx.Response(200, json={"message": {"content": text}})

    be = OllamaBackend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = be.generate([IMG], "check", RoiAudit, "sys")
    assert out.visibility == 2
    b = seen[0]
    assert b["model"] == "qwen3-vl:4b-instruct" and b["think"] is False and b["stream"] is False
    assert b["options"]["temperature"] == 0 and b["format"]["type"] == "object"
    assert b["messages"][0]["role"] == "system" and len(b["messages"][1]["images"]) == 1
    assert be.stats.calls == 1 and be.stats.invalid_json == 1 and be.stats.as_dict()["json_validity"] == 0.0


def test_ollama_400_retries_with_bigger_context_then_surfaces_error():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["options"]["num_ctx"])
        if body["options"]["num_ctx"] >= 16384:
            return httpx.Response(200, json={"message": {"content": ROI_JSON}})
        return httpx.Response(400, text="context length exceeded")

    be = OllamaBackend(num_ctx=8192, client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = be.generate([IMG], "x", RoiAudit)
    assert out.visibility == 2 and seen == [8192, 16384]  # retried once with a bigger window

    be2 = OllamaBackend(num_ctx=8192,
                        client=httpx.Client(transport=httpx.MockTransport(
                            lambda r: httpx.Response(400, text="some other bad request"))))
    with pytest.raises(VLMError, match="400"):
        be2.generate([IMG], "x", RoiAudit)


def test_ollama_missing_model_message():
    be = OllamaBackend(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    with pytest.raises(VLMError, match="ollama pull"):
        be.generate([IMG], "x", RoiAudit)


def test_ollama_warns_on_non_instruct_tag():
    with pytest.warns(UserWarning, match="thinking"):
        OllamaBackend(model="qwen3-vl:8b", client=httpx.Client())


def test_openai_compat_request_shape():
    calls = []

    class Msg:
        content = ROI_JSON

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            return Resp()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    be = OpenAICompatBackend(client=Client())
    assert be.generate([IMG, IMG], "go", RoiAudit).visibility == 2
    kw = calls[0]
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    content = kw["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 2 and content[-1]["type"] == "text"


def test_scripted_failure_raises_after_retry():
    be = ScriptedBackend(lambda p, s, n: "garbage")
    with pytest.raises(VLMError):
        be.generate([], "x", SceneAudit)
    assert be.stats.failures == 1
