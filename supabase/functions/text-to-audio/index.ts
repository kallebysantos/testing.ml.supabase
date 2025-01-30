// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import {
  PreTrainedTokenizer,
  RawAudio,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1";

import InferenceAPI from "inference-api";
const { Tensor, RawSession } = InferenceAPI;

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;
const MODEL_ID = "onnx-community/Kokoro-82M-ONNX";

const session = await RawSession.fromHuggingFace(MODEL_ID);

console.log(session);
/*
session {
  id: "6325400502544830693",
  input: [ "input_ids", "style", "speed" ],
  output: [ "waveform" ]
}
*/

Deno.serve(async (_) => {
  const tokenizer = await loadTokenizer();
  const { input_ids } = tokenizer("Hello");

  // Select voice style based on number of input tokens
  const num_tokens = Math.max(
    input_ids.dims.at(-1) - 2, // Without padding;
    0,
  );

  const voiceStyle = await loadVoiceStyle(num_tokens);

  const { waveform } = await session.run({
    input_ids,
    style: voiceStyle,
    speed: new Tensor("float32", [1], [1]),
  });

  const audio = new RawAudio(waveform.data, SAMPLE_RATE);
  console.log(waveform, audio);

  return new Response(await audio.toBlob(), {
    headers: {
      "Content-Type": "audio/wav",
    },
  });
});

async function loadVoiceStyle(num_tokens: number) {
  const voice_url =
    "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af_bella.bin?download=true";

  const voiceBuffer = await fetch(voice_url).then(async (res) =>
    await res.arrayBuffer()
  );

  const offset = num_tokens * STYLE_DIM;
  const voiceData = new Float32Array(voiceBuffer).slice(
    offset,
    offset + STYLE_DIM,
  );

  return new Tensor("float32", voiceData, [1, STYLE_DIM]);
}

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
