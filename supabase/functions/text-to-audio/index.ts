// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import {
  PreTrainedTokenizer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1";
import { KokoroTTS } from "https://cdn.jsdelivr.net/npm/kokoro-js@1.0.1/+esm";

import InferenceAPI from "inference-api";
const { RawSession } = InferenceAPI;

const MODEL_ID = "onnx-community/Kokoro-82M-ONNX";

const session = await RawSession.fromHuggingFace(MODEL_ID);
const tokenizer = await loadTokenizer();
// BUG: KokoroTTS.from_pretrained() can't load tokenizer, so we need to pass manually
const kokoro = new KokoroTTS(
  async (...args) => await session.run(...args),
  tokenizer,
);

Deno.serve(async (req) => {
  const params = new URL(req.url).searchParams;
  const text = params.get("text");
  const voice = params.get("voice") ?? "af_bella";

  const audio = await kokoro.generate(text, { voice });

  console.log("generated", audio);

  return new Response(await audio.toBlob(), {
    headers: {
      "Content-Type": "audio/wav",
    },
  });
});

async function loadTokenizer() {
  // BUG: invalid 'h' not JSON. That's why we need to manually fetch the assets
  // const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

  const tokenizerData = await fetch(
    "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/tokenizer.json?download=true",
  ).then(async (res) => await res.json());

  const tokenizerConfig = await fetch(
    "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/tokenizer_config.json?download=true",
  ).then(async (res) => await res.json());

  console.log(tokenizerConfig);

  return new PreTrainedTokenizer(tokenizerData, tokenizerConfig);
}
